# Index freshness: delta indexing

| Step | Chunks embedded | Chunks deleted | Chunks unchanged (skipped) | Chunks in index | New document retrievable |
|---|---|---|---|---|---|
| 1. sync, corpus unchanged | 0 | 0 | 965 | 965 | False |
| 2. new document added | 2 | 0 | 965 | 967 | True |
| 3. same document edited (one line) | 1 | 1 | 966 | 967 | True |
| 4. document deleted | 0 | 2 | 965 | 965 | False |

Probe question: "When was the harassment back-fill for audit finding F-4 run?". Only the changed document's chunks are embedded or deleted; the rest of the corpus is never re-embedded.
