## Use the smallest sufficient search

For a one-variable grid boundary, seek four evidence roles: requested baseline, one informative coarse grid point, adjacent failing point, and adjacent passing point. A corrected invalid call replaces its invalid predecessor; the invalid result is not evidence.

Call only requested grid values. As soon as adjacent grid points give one fail and one pass and all `LIMITS` are checked, stop all tools immediately. Do not repeat either point, test between them, or add confirmation beyond them.

For discrete candidates, test each under the same `TOOL_BASE`, reject failures, and rank survivors. For multi-case windows, refine only the cases controlling each boundary.
