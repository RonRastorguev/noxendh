# Change-set hashes: approved vs. current wording

On 2026-10-18 the wording of `PARAM_CHANGESET_S1.json` and `PARAM_CHANGESET_S2.json` was edited (source notes only:
references to our other projects removed). The parameter names and values are unchanged. Ron's go was given on the
files as they were then, so the sha256 values in `docs/NEEDS_RON.md` (R13) and in the apply logs are the APPROVED
hashes, not the current files':

| File | Approved sha256 (applied 2026-10-15) | Current sha256 (reworded) |
|---|---|---|
| PARAM_CHANGESET_S1.json | aa3c6176a16328585c4f9bda6afab962939b0c309e7385849d6ae058c449df18 | dccf33f60203425f7db97c0c6f4e3e10236576c2e9bacdf8b880fbfa8a211db1 |
| PARAM_CHANGESET_S2.json | bff26d4c114cfb8ab370b396573bed3d18533b27cad7e4840625a13a4fcc3d3e | d593a2bbcd9b6f0332d022ef32a3205af64433d5486a2af71fee3ce900ae1182 |

S3–S6 are unchanged: their files still hash to the approved values.
