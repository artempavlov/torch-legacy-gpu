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

Bootstrap regressions: keep `core.longpaths=true` on both source clone and
recursive submodule commands (Windows FlashAttention paths); install `libxml2`
before running the CUDA 10.2 installer. Prefer compatible binary dependencies,
do not install CFFI for modern torch (it rejects Python 3.13t), and respect
PyTorch 2.7's `setuptools<=75.8.2` limit.
Old Python installed via Conda must be used as the base of an isolated venv;
do not mix pip upgrades with Conda's distutils-owned packages (notably certifi).
Always pass an explicit build-host glibc platform to auditwheel repair. Older
auditwheel defaults to manylinux_2_5 and rejects the Ubuntu-built wheels.
Install pip's patchelf>=0.14.5 alongside auditwheel; Ubuntu 18.04's system
patchelf 0.9 and Ubuntu 22.04's 0.14.3 are too old for current auditwheel.
Upgrade pip inside the fresh smoke-test venv before installing repaired wheels.
Python 3.6 seeds pip 18.1; PEP 600 manylinux tags require pip 20.3 or later.
Before Linux torchaudio repair, unpack the wheel, add $ORIGIN to each bundled
ELF's RPATH, then repack with wheel to regenerate RECORD. Upstream CMake strips
the install RPATH; without this, auditwheel cannot resolve sibling libraries
such as libtorchaudio.so. Never exclude these bundled libraries from repair.
Release uploads retry transient HTTP/transport errors with bounded backoff.
Before retrying, delete only the exact filename's incomplete starter/open asset,
or accept an already uploaded asset only after matching its SHA-256 and size.
Keep the bundle manifest last; never mark a partially uploaded bundle complete.
