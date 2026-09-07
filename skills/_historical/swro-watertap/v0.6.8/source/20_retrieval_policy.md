## When to retrieve

Retrieve only when it adds missing structure:

- task wording ambiguous, domain framing needed;
- a tool argument name or supported option is uncertain;
- engineering interpretation, monitoring indicators, or mitigation logic is asked;
- species, units, or scaling interpretation need confirmation.

Skip retrieval when the question already specifies the needed inputs and procedure.

## Retrieval discipline

When used:

- extract only actionable facts; do not copy large blocks of retrieved text into the answer;
- distinguish definition-level facts from case-specific claims;
- treat retrieved numbers as guidance only, unless stated in the question;
- never manufacture hidden assumptions from retrieval;
- if retrieval is noisy or conflicting, fall back to question values plus tools.

Useful retrieval outputs: which variable to search, which tool to use, what the tool expects for
arguments and units, and what interpretation should accompany the result.
