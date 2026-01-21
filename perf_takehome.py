"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
"""

from collections import defaultdict
from dataclasses import dataclass
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)

# Number of cores the kernel partitions across. Must match Machine.n_cores when executing.
KERNEL_CORES = 4


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.const_loads = []

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.const_loads.append((addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized kernel builder using vectorized operations and multi-core
        partitioning with a dependency-aware VLIW scheduler.
        """
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.const_loads = []

        @dataclass(frozen=True)
        class MicroOp:
            engine: str
            slot: tuple
            reads: tuple[int, ...]
            writes: tuple[int, ...]

        def vec_range(addr: int) -> tuple[int, ...]:
            return tuple(range(addr, addr + VLEN))

        def add_op(ops, engine, slot, reads=(), writes=()):
            ops.append(MicroOp(engine, slot, tuple(reads), tuple(writes)))

        def schedule_ops(ops):
            last_writer = {}
            last_reader = {}
            deps = [set() for _ in ops]
            users = [set() for _ in ops]

            for op_id, op in enumerate(ops):
                cur_deps = set()
                for addr in op.reads:
                    if addr in last_writer:
                        cur_deps.add(last_writer[addr])
                for addr in op.writes:
                    if addr in last_writer:
                        cur_deps.add(last_writer[addr])
                    if addr in last_reader:
                        cur_deps.add(last_reader[addr])
                deps[op_id] = cur_deps
                for dep in cur_deps:
                    users[dep].add(op_id)
                for addr in op.reads:
                    last_reader[addr] = op_id
                for addr in op.writes:
                    last_writer[addr] = op_id
                    if addr in last_reader:
                        del last_reader[addr]

            indegree = [len(dep) for dep in deps]
            ready = [op_id for op_id, degree in enumerate(indegree) if degree == 0]
            scheduled = 0
            bundles = []

            while scheduled < len(ops):
                bundle = defaultdict(list)
                used = defaultdict(int)
                next_ready = []
                scheduled_ids = []

                for op_id in ready:
                    op = ops[op_id]
                    limit = SLOT_LIMITS.get(op.engine, 0)
                    if used[op.engine] < limit:
                        bundle[op.engine].append(op.slot)
                        used[op.engine] += 1
                        scheduled_ids.append(op_id)
                    else:
                        next_ready.append(op_id)

                if not scheduled_ids:
                    raise RuntimeError("Scheduling stalled with no available slots.")

                for op_id in scheduled_ids:
                    scheduled += 1
                    for user in users[op_id]:
                        indegree[user] -= 1
                        if indegree[user] == 0:
                            next_ready.append(user)

                bundles.append(dict(bundle))
                ready = next_ready

            return bundles

        ops = []

        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        core_id = self.alloc_scratch("core_id")
        per_core = self.alloc_scratch("per_core")
        base_offset = self.alloc_scratch("base_offset")
        base_idx = self.alloc_scratch("base_idx")
        base_val = self.alloc_scratch("base_val")

        zero_const = self.scratch_const(0, "const_zero")
        two_const = self.scratch_const(2, "const_two")
        n_cores_const = self.scratch_const(KERNEL_CORES, "const_n_cores")

        hash_val_consts = [self.scratch_const(val) for _, val, _, _, _ in HASH_STAGES]
        hash_shift_consts = [
            self.scratch_const(shift) for _, _, _, _, shift in HASH_STAGES
        ]

        for i in range(len(init_vars)):
            self.scratch_const(i, f"const_init_{i}")

        if batch_size % KERNEL_CORES != 0:
            raise ValueError("batch_size must be divisible by KERNEL_CORES.")
        if (batch_size // KERNEL_CORES) % VLEN != 0:
            raise ValueError("Per-core batch size must be divisible by VLEN.")
        vectors_per_core = batch_size // (KERNEL_CORES * VLEN)
        vec_offsets = [v * VLEN for v in range(vectors_per_core)]
        offset_consts = [self.scratch_const(offset) for offset in vec_offsets]

        for addr, val in self.const_loads:
            add_op(ops, "load", ("const", addr, val), writes=(addr,))

        add_op(ops, "flow", ("coreid", core_id), writes=(core_id,))

        for i, v in enumerate(init_vars):
            const_i = self.const_map[i]
            add_op(
                ops,
                "load",
                ("load", self.scratch[v], const_i),
                reads=(const_i,),
                writes=(self.scratch[v],),
            )

        add_op(
            ops,
            "alu",
            ("//", per_core, self.scratch["batch_size"], n_cores_const),
            reads=(self.scratch["batch_size"], n_cores_const),
            writes=(per_core,),
        )
        add_op(
            ops,
            "alu",
            ("*", base_offset, core_id, per_core),
            reads=(core_id, per_core),
            writes=(base_offset,),
        )
        add_op(
            ops,
            "alu",
            ("+", base_idx, self.scratch["inp_indices_p"], base_offset),
            reads=(self.scratch["inp_indices_p"], base_offset),
            writes=(base_idx,),
        )
        add_op(
            ops,
            "alu",
            ("+", base_val, self.scratch["inp_values_p"], base_offset),
            reads=(self.scratch["inp_values_p"], base_offset),
            writes=(base_val,),
        )

        forest_base_vec = self.alloc_scratch("forest_base_vec", VLEN)
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", VLEN)
        zero_vec = self.alloc_scratch("zero_vec", VLEN)
        two_vec = self.alloc_scratch("two_vec", VLEN)
        hash_val_vecs = [
            self.alloc_scratch(f"hash_val_{i}", VLEN) for i in range(len(HASH_STAGES))
        ]
        hash_shift_vecs = [
            self.alloc_scratch(f"hash_shift_{i}", VLEN)
            for i in range(len(HASH_STAGES))
        ]

        add_op(
            ops,
            "valu",
            ("vbroadcast", forest_base_vec, self.scratch["forest_values_p"]),
            reads=(self.scratch["forest_values_p"],),
            writes=vec_range(forest_base_vec),
        )
        add_op(
            ops,
            "valu",
            ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"]),
            reads=(self.scratch["n_nodes"],),
            writes=vec_range(n_nodes_vec),
        )
        add_op(
            ops,
            "valu",
            ("vbroadcast", zero_vec, zero_const),
            reads=(zero_const,),
            writes=vec_range(zero_vec),
        )
        add_op(
            ops,
            "valu",
            ("vbroadcast", two_vec, two_const),
            reads=(two_const,),
            writes=vec_range(two_vec),
        )

        for i, const_addr in enumerate(hash_val_consts):
            add_op(
                ops,
                "valu",
                ("vbroadcast", hash_val_vecs[i], const_addr),
                reads=(const_addr,),
                writes=vec_range(hash_val_vecs[i]),
            )
        for i, const_addr in enumerate(hash_shift_consts):
            add_op(
                ops,
                "valu",
                ("vbroadcast", hash_shift_vecs[i], const_addr),
                reads=(const_addr,),
                writes=vec_range(hash_shift_vecs[i]),
            )

        addr_idx_addrs = []
        addr_val_addrs = []
        vector_scratch = []

        for vec_i, offset_const in enumerate(offset_consts):
            addr_idx = self.alloc_scratch(f"addr_idx_{vec_i}")
            addr_val = self.alloc_scratch(f"addr_val_{vec_i}")
            addr_idx_addrs.append(addr_idx)
            addr_val_addrs.append(addr_val)
            add_op(
                ops,
                "alu",
                ("+", addr_idx, base_idx, offset_const),
                reads=(base_idx, offset_const),
                writes=(addr_idx,),
            )
            add_op(
                ops,
                "alu",
                ("+", addr_val, base_val, offset_const),
                reads=(base_val, offset_const),
                writes=(addr_val,),
            )

            idx_vec = self.alloc_scratch(f"idx_vec_{vec_i}", VLEN)
            val_vec = self.alloc_scratch(f"val_vec_{vec_i}", VLEN)
            addr_vec = self.alloc_scratch(f"addr_vec_{vec_i}", VLEN)
            node_vec = self.alloc_scratch(f"node_vec_{vec_i}", VLEN)
            tmp1_vec = self.alloc_scratch(f"tmp1_vec_{vec_i}", VLEN)
            tmp2_vec = self.alloc_scratch(f"tmp2_vec_{vec_i}", VLEN)
            tmp3_vec = self.alloc_scratch(f"tmp3_vec_{vec_i}", VLEN)

            vector_scratch.append(
                {
                    "idx": idx_vec,
                    "val": val_vec,
                    "addr": addr_vec,
                    "node": node_vec,
                    "tmp1": tmp1_vec,
                    "tmp2": tmp2_vec,
                    "tmp3": tmp3_vec,
                }
            )

            add_op(
                ops,
                "load",
                ("vload", idx_vec, addr_idx),
                reads=(addr_idx,),
                writes=vec_range(idx_vec),
            )
            add_op(
                ops,
                "load",
                ("vload", val_vec, addr_val),
                reads=(addr_val,),
                writes=vec_range(val_vec),
            )

        for _round in range(rounds):
            for vec in vector_scratch:
                idx_vec = vec["idx"]
                val_vec = vec["val"]
                addr_vec = vec["addr"]
                node_vec = vec["node"]
                tmp1_vec = vec["tmp1"]
                tmp2_vec = vec["tmp2"]
                tmp3_vec = vec["tmp3"]

                add_op(
                    ops,
                    "valu",
                    ("+", addr_vec, idx_vec, forest_base_vec),
                    reads=vec_range(idx_vec) + vec_range(forest_base_vec),
                    writes=vec_range(addr_vec),
                )

                for offset in range(VLEN):
                    add_op(
                        ops,
                        "load",
                        ("load_offset", node_vec, addr_vec, offset),
                        reads=(addr_vec + offset,),
                        writes=(node_vec + offset,),
                    )

                add_op(
                    ops,
                    "valu",
                    ("^", val_vec, val_vec, node_vec),
                    reads=vec_range(val_vec) + vec_range(node_vec),
                    writes=vec_range(val_vec),
                )

                for stage, (op1, _val1, op2, op3, _shift) in enumerate(HASH_STAGES):
                    add_op(
                        ops,
                        "valu",
                        (op1, tmp1_vec, val_vec, hash_val_vecs[stage]),
                        reads=vec_range(val_vec) + vec_range(hash_val_vecs[stage]),
                        writes=vec_range(tmp1_vec),
                    )
                    add_op(
                        ops,
                        "valu",
                        (op3, tmp2_vec, val_vec, hash_shift_vecs[stage]),
                        reads=vec_range(val_vec) + vec_range(hash_shift_vecs[stage]),
                        writes=vec_range(tmp2_vec),
                    )
                    add_op(
                        ops,
                        "valu",
                        (op2, val_vec, tmp1_vec, tmp2_vec),
                        reads=vec_range(tmp1_vec) + vec_range(tmp2_vec),
                        writes=vec_range(val_vec),
                    )

                add_op(
                    ops,
                    "valu",
                    ("%", tmp1_vec, val_vec, two_vec),
                    reads=vec_range(val_vec) + vec_range(two_vec),
                    writes=vec_range(tmp1_vec),
                )
                add_op(
                    ops,
                    "valu",
                    ("==", tmp2_vec, tmp1_vec, zero_vec),
                    reads=vec_range(tmp1_vec) + vec_range(zero_vec),
                    writes=vec_range(tmp2_vec),
                )
                add_op(
                    ops,
                    "valu",
                    ("-", tmp3_vec, two_vec, tmp2_vec),
                    reads=vec_range(two_vec) + vec_range(tmp2_vec),
                    writes=vec_range(tmp3_vec),
                )
                add_op(
                    ops,
                    "valu",
                    ("*", tmp1_vec, idx_vec, two_vec),
                    reads=vec_range(idx_vec) + vec_range(two_vec),
                    writes=vec_range(tmp1_vec),
                )
                add_op(
                    ops,
                    "valu",
                    ("+", idx_vec, tmp1_vec, tmp3_vec),
                    reads=vec_range(tmp1_vec) + vec_range(tmp3_vec),
                    writes=vec_range(idx_vec),
                )
                add_op(
                    ops,
                    "valu",
                    ("<", tmp2_vec, idx_vec, n_nodes_vec),
                    reads=vec_range(idx_vec) + vec_range(n_nodes_vec),
                    writes=vec_range(tmp2_vec),
                )
                add_op(
                    ops,
                    "valu",
                    ("*", idx_vec, idx_vec, tmp2_vec),
                    reads=vec_range(idx_vec) + vec_range(tmp2_vec),
                    writes=vec_range(idx_vec),
                )

        for vec_i, vec in enumerate(vector_scratch):
            add_op(
                ops,
                "store",
                ("vstore", addr_idx_addrs[vec_i], vec["idx"]),
                reads=(addr_idx_addrs[vec_i],) + vec_range(vec["idx"]),
            )
            add_op(
                ops,
                "store",
                ("vstore", addr_val_addrs[vec_i], vec["val"]),
                reads=(addr_val_addrs[vec_i],) + vec_range(vec["val"]),
            )

        self.instrs = schedule_ops(ops)

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=KERNEL_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
