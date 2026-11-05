# Build automation requirements

## 1. Problem
Official PyTorch wheels drop older NVIDIA compute capabilities.

## 2. Goals and non-goals
Automate torch/torchvision/torchaudio releases for Kepler and newer. Preserve old
compatible combinations when modern sources/toolkits no longer support a GPU.
No macOS, Fermi, Jetson/JetPack, service deployment or upstream workflow fork.

## 3. Workflows
Six-hour upstream polling; stable-release discovery; compatibility planning;
native hosted compilation; repair; smoke checks; draft upload; complete publication.
Manual dispatch can select a release or individual bundle.

## 4. Functional requirements
Linux x86_64, Windows x86_64, Linux ARM64 SBSA where feasible. All Python minors
supported by each source baseline. Companion versions follow declared torch
requirements. Completed artifacts survive subsequent failures and are skipped.

## 5. Nonfunctional requirements
Visible diagnostics, exact source commits, SHA-256 checksums, architecture flags,
honest verification status, bounded parallelism, no paid infrastructure by default.

## 6. Constraints
Prefer upstream workflow reuse without copying. Unchanged upstream reusable
workflow cannot supply external checkout/architecture overrides or hosted runners;
therefore use upstream packaging entrypoints from official source tags instead.
Compilation has a six-hour hosted job limit. No NVIDIA GPU execution test is
possible on standard hosted runners. Old CUDA/compilers/Python constrain coverage.

## 7. Edge cases
Draft/prerelease ignored; interrupted uploads retried; missing companion fails;
unknown compatibility table fails closed; partial lane remains draft; wheel size
limit fails before publication; new configuration needs a release revision bump.

## 8. Risks and unresolved evidence
Initial Kepler recipes are candidates, not established last-possible baselines.
Modern upstream sources may eventually require a newer toolkit. Hosted runner
memory/disk/time can be insufficient. ARM64 SBSA is not Jetson. Real hardware
verification remains separate from successful compilation/import/target inspection.

## 9. Implementation plan
Configuration -> planner -> native source builds -> repaired bundle verification
-> manifest-last drafts -> complete release publication -> local and hosted checks.

## 10. Acceptance criteria
Unit tests and workflow lint pass. Real upstream metadata yields a valid matrix.
Published groups contain all three packages for every expected platform/ABI, with
manifests and smoke evidence. No published group claims GPU execution evidence
without a hardware test. Unsupported future combinations fail visibly.

## 11. Next-agent prompt
Read AGENTS.md and README. Inspect current Actions runs before changing policy.
Reproduce failed source build boundaries, distinguish runner limits from actual
architecture incompatibility, and update tests and documentation with evidence.
Never declare a recipe proven without a completed build, nor call a frozen
candidate the newest possible compatible version without verifying that claim.
