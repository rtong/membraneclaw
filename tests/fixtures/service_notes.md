<!--
SYNTHETIC TEST FIXTURE — NOT DOCUMENTATION.

Every fact below is invented. There is no ingest service, no port 7431, and no
`membrane-results` bucket anywhere in this project. These values exist so
smoke_test.py can prove RAG retrieved *this file* rather than answering from
model memory — which requires facts the model could not otherwise know.

Do not point AGENT_FILES at this file. Doing so attaches fabricated "service
notes" to every production request, and the agent will state them as fact.
-->

# Fixture Service Notes (synthetic)

The ingest service listens on port 7431 and exposes a single POST endpoint at /v1/ingest.

Retries use exponential backoff starting at 250ms with a cap of 30 seconds.
The maximum accepted payload size is 12 MiB; larger uploads are rejected with HTTP 413.

Batch jobs run nightly at 02:15 UTC and write results to the `membrane-results` bucket.
