# D7 intake placeholder

## Current decision

D7 will first be accepted in the same source format as D1-D6. No extra worksheet, R0/R2 label,
retrieval evidence field, or filename convention is imposed before the real delivery is inspected.

The current code only reserves an import entry for Excel workbooks placed under:

```text
benchmarks/Datasets Harness/D7/
```

Each workbook is expected to expose the same four logical sheet roles already supported by D1-D6:

1. Question
2. Gold or stepwise answer
3. Trajectory/evaluation rubric
4. Tool-efficiency rubric

Existing Chinese and English sheet aliases are reused by the normal benchmark importer.

## Post-delivery workflow

After the D7 files arrive:

1. copy the source files unchanged into the D7 directory;
2. inspect filenames, sheet names, column layout, task families, rubrics, and question content;
3. run the normal importer and fix only compatibility issues discovered from the real structure;
4. audit every case for whether the question is self-contained or genuinely needs the frozen RAG corpus;
5. create routing labels and evidence metadata as a derived sidecar or normalized-data view, without
   asking the benchmark author to redesign the original workbooks;
6. decide whether D7 already contains suitable R0 controls or whether R0 should be sampled from
   D1-D6 for the combined Router evaluation;
7. only then add the combined D1-D7 set and the final formal experiment commands.

## Deferred decisions

The following are intentionally not fixed until the real D7 structure is available:

- the R0/R2 annotation storage format;
- the number and balance of R0/R2 cases;
- the exact RAG evidence mapping;
- whether D7 supplies its own R0 controls;
- the combined D1-D7 benchmark configuration;
- the final D7 smoke-test case IDs.

This avoids changing colleague-authored workbooks based on assumptions that may not match the delivered
data.
