# MembraneClaw Service Notes

The ingest service listens on port 7431 and exposes a single POST endpoint at /v1/ingest.

Retries use exponential backoff starting at 250ms with a cap of 30 seconds.
The maximum accepted payload size is 12 MiB; larger uploads are rejected with HTTP 413.

Batch jobs run nightly at 02:15 UTC and write results to the `membrane-results` bucket.
