# Legacy four-level models

The files in this folder belong to the original four-level OELM API and are not
loaded by the two-level feature branch.

The active deployment resources are in `binary_models/`. The Docker image copies
only `app.py`, `requirements.txt`, and `binary_models/`, so these legacy files
are not included in the Hugging Face image.
