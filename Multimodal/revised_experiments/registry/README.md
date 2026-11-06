# Immutable run registry

Every v2 run must receive a unique output directory and a `run_record.json`
conforming to `schemas/run_record.schema.json`. Per-segment and per-call outputs
must conform to `schemas/prediction_record.schema.json`.

Historical v1 artifacts are never rewritten into this format as if their missing
fields were known. The v1 freeze records unknown lineage explicitly.

Before training, copy and complete `templates/experiment_card.yaml`; its hash is
then referenced by every run in that comparison family.
