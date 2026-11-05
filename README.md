# torch-legacy-gpu

[![Automation checks](https://github.com/artempavlov/torch-legacy-gpu/actions/workflows/check.yml/badge.svg)](https://github.com/artempavlov/torch-legacy-gpu/actions/workflows/check.yml)
[![CUDA builds](https://github.com/artempavlov/torch-legacy-gpu/actions/workflows/build.yml/badge.svg)](https://github.com/artempavlov/torch-legacy-gpu/actions/workflows/build.yml)

Unofficial **torch, torchvision and torchaudio** wheels for older NVIDIA GPUs.
GitHub Actions checks upstream stable releases every six hours and builds on
GitHub-hosted Linux x86_64, Linux ARM64 and Windows x86_64 runners.

**Bootstrap status: build recipes, not a promise of working binaries.** A lane is
only available when its complete release appears in [Releases](https://github.com/artempavlov/torch-legacy-gpu/releases).
Compiler support, successful imports and actual GPU execution are different things.
Hosted runners have no NVIDIA GPU; the latter requires testing on your hardware.

## Compatibility policy

| Lane | Compute capabilities | CUDA | PyTorch | Platforms |
| --- | --- | --- | --- | --- |
| kepler30 | 3.0 | 10.2 | 1.6.0 candidate baseline | Linux/Windows x86_64 |
| kepler | 3.5, 3.7 | 11.8 | 2.7.1 candidate baseline | Linux/Windows x86_64 |
| maxwell | 5.0, 5.2 | 12.6 | New upstream releases | Linux/Windows x86_64 |
| pascal | 6.0, 6.1 | 12.6 | New upstream releases | Linux/Windows x86_64 |
| volta | 7.0 | 12.6 | New upstream releases | Linux/Windows x86_64; Linux ARM64 SBSA |
| modern | 7.5, 8.0, 8.6, 8.9, 9.0, 10.0, 12.0 + PTX | 13.2 | New upstream releases | Linux/Windows x86_64 |
| modern ARM | 8.0, 9.0, 10.0, 11.0, 12.0 + PTX | 13.2 | New upstream releases | Linux ARM64 SBSA |

All CPython minors in the upstream release compatibility row are planned,
including free-threaded Python from 3.13 where upstream lists support. Frozen
lanes retain their historical Python versions (3.6–3.8 and 3.9–3.13/3.13t).
Companion versions are selected from their actual PyPI `Requires-Dist: torch`
version constraints, **not** guessed from matching minor version numbers.
Unversioned `torch` requirements are not accepted as compatibility evidence;
the oldest unversioned packages use explicit historical pins (vision 0.7.0,
audio 0.6.0 for torch 1.6.0).

Kepler cannot be restored to modern CUDA by an architecture flag: CUDA 11
removed sm_30, CUDA 12 removed sm_35/sm_37, and CUDA 13 removed compilation for
architectures older than 7.5. The old baselines are conservative candidates,
not a proven claim that no newer source can possibly build. Successful old
releases remain available when new ones fail. Future source incompatibility
fails visibly and requires updating the compatibility policy; it is not hidden
by silently substituting an older PyTorch under a new version number.

**Not covered:** Fermi, 32-bit ARM, Jetson/JetPack (sm_32, sm_53, sm_62, sm_72,
sm_87 and their platform-specific CUDA libraries), macOS, ROCm, libtorch archives,
and arbitrary future architectures unknown to the pinned toolkits. SBSA wheels
are not Jetson wheels. PTX provides a forward-compatibility fallback, not a
promise that every newer GPU is supported.

## Install

1. Find your GPU's compute capability in [NVIDIA's legacy GPU list](https://developer.nvidia.com/cuda-legacy-gpus).
2. Open the matching lane release. Download **all three wheels** for the same
   Python ABI, OS, CPU architecture and local version suffix.
3. In an isolated environment, install the downloaded files together:

   ```sh
   python -m pip install ./torch-*.whl ./torchvision-*.whl ./torchaudio-*.whl
   python -m pip check
   ```

Use explicit filenames in PowerShell (it does not expand native-command globs
the same way as a POSIX shell). Do not mix lanes or substitute official torch
wheels. Manifests record SHA-256 hashes, exact upstream commits, compiled target
flags, disabled features, and verification results. Version suffixes identify
the toolkit, architecture family and Python ABI; package names stay unchanged.

An NVIDIA driver compatible with the chosen toolkit **and** your GPU is still
required. Wheels bundle redistributable runtime dependencies, never the driver.
Linux wheels carry the ABI tag determined by auditwheel; building on Ubuntu
22.04/24.04 does not promise manylinux2014 compatibility.

## What is reused from upstream?

No upstream workflow is copied here. The official PyTorch reusable workflow
assumes PyTorch's own checkout, internal runner labels, packaging images and
release infrastructure; it does not expose the source/architecture overrides
needed by this repository. Calling it unchanged would build the wrong checkout
or leave jobs queued for unavailable runners.

The small local workflow instead checks out official release tags recursively
and invokes **their own build backends**: `pip wheel` for modern PEP 517 PyTorch,
or the original `setup.py bdist_wheel` for older versions and companion libraries.
The only source edit is a checked, one-line sm_30 extension allowlist addition in
PyTorch 1.6.0. CUDA API/compiler checks and kernels are not patched away.

For legacy coverage and hosted-runner resource limits these are **reduced-feature
builds**, not replacements for every feature of official PyTorch wheels.
cuDNN, distributed/NCCL, FlashAttention, memory-efficient attention, cuSPARSELt,
cuFile, Kineto, MKLDNN, NNPACK, FFmpeg, SoX, Kaldi and NVJPEG are disabled.
Image codecs depend on the platform's available development libraries. The
old audio 0.6 Linux extension still requires system SoX (bundled by repair);
`BUILD_SOX=0` disables building its vendored SoX, not that old API dependency.

## Automation and recovery

- `config.json` defines lanes. Discovery starts at PyTorch 2.15.0; historical
  Kepler baselines are also built once. Prereleases/drafts are ignored.
- Scheduled runs and pushes to build automation discover pending bundles.
  Manual dispatch accepts a stable `torch` version and/or exact `only` bundle ID.
  Download the `plan` artifact to find IDs and the full intended matrix.
- Compilation uses at most two concurrent compiler processes. Architecture
  families are separated to reduce memory and wheel size. Linux ccache is saved
  even after a controlled timeout where possible; cache retention is not guaranteed.
- Successful bundles upload three repaired wheels followed by a JSON commit
  marker to a draft release. Only a complete lane/platform/Python group is
  published. Failures remain visible; drafts are not advertised as releases.
- Reruns skip complete bundles and retry unfinished ones. Published assets are
  immutable; bump `revision` to change a published recipe. Old releases are not deleted.
- GitHub's six-hour job limit, queue quotas, disk/RAM, cache eviction and the
  2 GiB asset limit can prevent large builds. There is no claim that every
  upstream release fits free hosted runners. Such jobs fail rather than publish
  CPU-only or incomplete wheels. A compatible paid/self-hosted runner may be
  necessary in the future; none is configured or charged by this project.

The planner has read-only permissions. Only build publication/finalization can
write releases. No personal access token, external storage or PyPI credentials
are needed. Pull requests run checks only, not privileged release builds.

## Develop

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/plan.py --offline
```

`--offline` uses representative synthetic fixture data, not real built wheels.
`actionlint` checks both workflows. Build scripts retain Python 3.6 syntax for
the oldest lane; discovery/publication use Python 3.13.

## Sources and licenses

- [PyTorch release compatibility matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)
- [Official reusable Linux workflow](https://github.com/pytorch/pytorch/blob/main/.github/workflows/_binary-build-linux.yml)
- [CUDA 12 release notes](https://docs.nvidia.com/cuda/archive/12.0.0/cuda-toolkit-release-notes/index.html)
- [NVIDIA architecture support policy](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/)
- [GitHub Actions limits](https://docs.github.com/en/actions/reference/limits)

Repository automation: MIT. PyTorch, companion libraries, third-party code and
CUDA redistributables retain their upstream licenses. Their wheel license files
are not removed; this repository does not relicense NVIDIA components.
