## When to retrieve

Retrieve only when it adds missing structure:

- the task wording is ambiguous and domain framing is needed;
- a tool argument name or supported option is uncertain;
- engineering interpretation, monitoring indicators, or mitigation logic is asked;
- species, units, or scaling interpretation need confirmation.

Skip retrieval when the question already specifies the needed inputs and procedure.

## Retrieval discipline

When retrieval is used:

- extract only actionable facts; do not copy large blocks of retrieved text into the answer;
- distinguish definition-level facts from case-specific claims;
- treat retrieved numeric values as guidance only, unless they are explicitly stated in the question;
- never manufacture hidden assumptions from retrieval;
- if retrieval is noisy or conflicting, fall back to question values plus tools.

Useful retrieval outputs are: which variable to search, which tool to use, what the tool expects for
arguments and units, and what engineering interpretation should accompany the result.