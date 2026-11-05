# torch-legacy-gpu

CI-only project. No application, web service, system deployment, or private
development submodules. Control-plane Python >=3.11; build.py and smoke.py must
also parse and run on Python 3.6. Runtime dependency: packaging (requirements.txt).

- `config.json`: compatibility lanes and immutable release revision.
- `scripts/plan.py`: upstream discovery, Python/companion compatibility, matrix.
- `scripts/build.py`, `linux.sh`: upstream source builds, repair and smoke tests.
- `scripts/publish.py`: three-wheel validation, manifest-last draft publication.
- `.github/workflows/build.yml`: six-hour polling, native hosted builds.

Checks: `python -m unittest discover -s tests -v`,
`python scripts/plan.py --offline`, `actionlint`.
Use `.tmp/` for temporary files. Do not vendor upstream workflows. Do not claim
hardware support from successful compilation alone. Never publish an incomplete
bundle, silently downgrade a source version, or replace an official NVIDIA driver.
Changes to wheel contents/configuration require a revision bump after publication.
