# Anthropic's Original Performance Take-Home

This repo contains a version of Anthropic's original performance take-home, before Claude Opus 4.5 started doing better than humans given only 2 hours.

Now you can try to beat Claude Opus 4.5 given unlimited time!

## Performance benchmarks 

measured in clock cycles from the simulated machine:

- **2164 cycles**: Claude Opus 4 after many hours in the test-time compute harness
- **1790 cycles**: Claude Opus 4.5 in a casual Claude Code session, approximately matching the best human performance in 2 hours
- **1579 cycles**: Claude Opus 4.5 after 2 hours in our test-time compute harness
- **1548 cycles**: Claude Sonnet 4.5 after many more than 2 hours of test-time compute
- **1487 cycles**: Claude Opus 4.5 after 11.5 hours in the harness
- **1363 cycles**: Claude Opus 4.5 in an improved test time compute harness
- **617 cycles**: This repo’s vectorized multi-core kernel (local run of `tests/submission_tests.py -q`)

If you optimize below 1487 cycles, beating Claude Opus 4.5's best performance at launch, email us at performance-recruiting@anthropic.com with your code (and ideally a resume) so we can be appropriately impressed and perhaps discuss interviewing.

Run `python tests/submission_tests.py` to see which thresholds you pass.

## Documentation

- `EXPLANATION.md` provides a junior-friendly, Feynman-style walkthrough of the
  simulator, kernel, and optimization rationale.
- `NOTES.md` logs per-run notes and environment observations for continuity
  across sessions.
