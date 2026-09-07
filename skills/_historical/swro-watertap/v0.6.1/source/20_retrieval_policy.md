## When to retrieve

Run retrieval only when it adds missing structure. Typical reasons:

- the task wording is ambiguous and domain framing is needed;
- a tool argument name or supported option is uncertain;
- the question asks for an engineering interpretation, monitoring indicator, or mitigation logic;
- species, units, or scaling interpretation need confirmation.

Skip retrieval when the question already fully specifies the needed inputs and procedure.

After retrieval, keep only the few facts that change the execution plan. Do not copy large blocks of
retrieved text into the answer.

## Retrieval discipline

When retrieval is used:

- extract only actionable facts;
- distinguish definition-level facts from case-specific claims;
- treat retrieved numeric values as guidance only unless they are explicitly part of the question;
- do not manufacture hidden assumptions from retrieval;
- if retrieval is noisy or conflicting, fall back to question values plus tools.

Useful retrieval outputs are things like:

- which variable is worth searching;
- which tool is appropriate;
- what the tool expects for arguments and units;
- what engineering interpretation should accompany the computed result.
