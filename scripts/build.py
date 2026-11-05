"""Run official source build backends, then repair and test the resulting wheels.

Kept Python 3.6 compatible for the oldest lane. No GitHub credentials are needed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile


def run(*args, **kwargs):
    print("+", " ".join(str(a) for a in args), flush=True)
    return subprocess.check_call([str(a) for a in args], **kwargs)


def pip(*args, **kwargs):
    return run(sys.executable, "-m", "pip", *args, **kwargs)


def checkout(repository, version, destination):
    # -c is inherited by recursive Git processes, unlike a parent repo's local
    # config. FlashAttention's nested paths exceed Windows' default 260 chars.
    run("git", "-c", "core.longpaths=true", "clone", "--depth", "1", "--branch", "v" + version, "https://github.com/pytorch/" + repository + ".git", destination)
    run("git", "-c", "core.longpaths=true", "submodule", "update", "--init", "--recursive", "--depth", "1", cwd=str(destination))
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(destination)).decode().strip()


def patch_sm30(source):
    # Upstream 1.6's extension allowlist excludes 3.0 even though CUDA 10.2
    # accepts it. Change only this allowlist, not kernels or CUDA API checks.
    path = source / "torch/utils/cpp_extension.py"
    text = path.read_text()
    old = "supported_arches = ['3.5'"
    if text.count(old) != 1:
        raise RuntimeError("sm_30 allowlist patch no longer applies cleanly")
    path.write_text(text.replace(old, "supported_arches = ['3.0', '3.5'"))


def metadata(path):
    with zipfile.ZipFile(str(path)) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        return archive.read(name).decode()


def manylinux_platform():
    # Python 3.6's platform.libc_ver scans the interpreter executable, which
    # reports Conda's build baseline rather than this container's running libc.
    try:
        libc, version = os.confstr("CS_GNU_LIBC_VERSION").split()
    except (AttributeError, OSError, ValueError):
        raise RuntimeError("Cannot determine the build host's glibc ABI")
    parts = version.split(".")
    if libc != "glibc" or len(parts) < 2 or not all(p.isdigit() for p in parts[:2]):
        raise RuntimeError("Cannot determine the build host's glibc ABI")
    return "manylinux_" + "_".join(parts[:2]) + "_" + platform.machine()


def prepare_audio_wheel(wheel, scratch):
    # Upstream CMake removes the install RPATH. Its sibling DSOs then look
    # external to auditwheel, even though they are already inside this wheel.
    unpacked = scratch / "unpacked"
    packed = scratch / "packed"
    unpacked.mkdir(parents=True)
    packed.mkdir()
    run(sys.executable, "-m", "wheel", "unpack", "--dest", unpacked, wheel)
    package = next(unpacked.iterdir())
    for path in package.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                is_elf = stream.read(4) == b"\x7fELF"
            if is_elf:
                run("patchelf", "--add-rpath", "$ORIGIN", path)
    # Rebuild RECORD after changing the ELF files; do not edit a zip in place.
    run(sys.executable, "-m", "wheel", "pack", "--dest-dir", packed, package)
    return next(packed.glob("*.whl"))


def repair(wheel, out, cuda):
    if sys.platform == "win32":
        # Don't vendor NVIDIA's driver: it belongs to the user's system.
        run(sys.executable, "-m", "delvewheel", "repair", "--add-path", str(cuda / "bin"),
            "--exclude", "nvcuda.dll", "--wheel-dir", out, wheel)
    else:
        # CUDA libraries are redistributable runtime dependencies. libcuda is
        # deliberately excluded (driver). Older auditwheel defaults to glibc
        # 2.5, which rejects wheels built on Ubuntu 18.04 and later. Request the
        # build host's conservative baseline; auditwheel still checks symbols
        # and may additionally report an older ABI when the wheel supports it.
        exclude = ["--exclude", "libcuda.so.1"]
        if not wheel.name.startswith("torch-"):
            # These are supplied by the matching torch wheel, not by Linux.
            for library in ("libtorch.so", "libtorch_cpu.so", "libtorch_cuda.so", "libtorch_python.so", "libc10.so", "libc10_cuda.so"):
                exclude.extend(["--exclude", library])
        command = [sys.executable, "-m", "auditwheel", "repair", "--plat", manylinux_platform()] + exclude + ["--wheel-dir", out]
        if wheel.name.startswith("torchaudio-"):
            with tempfile.TemporaryDirectory(dir=str(out)) as scratch:
                prepared = prepare_audio_wheel(wheel, Path(scratch))
                run(*command, prepared)
        else:
            run(*command, wheel)


def create_smoke_environment(directory):
    run(sys.executable, "-m", "venv", directory)
    python = directory / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    # Python 3.6 seeds pip 18.1, which cannot install PEP 600 manylinux tags.
    # pip selects the newest release compatible with the interpreter.
    run(python, "-m", "pip", "install", "--upgrade", "pip")
    return python


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", default="dist")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    root = Path.cwd().resolve()
    work = root / "work"
    out = Path(args.output).resolve()
    work.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    cuda = Path(os.environ.get("CUDA_PATH", "/usr/local/cuda"))
    env = os.environ.copy()
    env.update({"CUDA_HOME": str(cuda), "CUDA_PATH": str(cuda),
                # Also applies to pip's isolated PEP 517 build environment.
                "PIP_PREFER_BINARY": "1",
                "TORCH_CUDA_ARCH_LIST": ";".join(spec["architectures"]),
                "MAX_JOBS": "2", "CMAKE_BUILD_PARALLEL_LEVEL": "2", "CMAKE_GENERATOR": "Ninja",
                "USE_CUDA": "1", "FORCE_CUDA": "1", "BUILD_TEST": "0",
                "USE_DISTRIBUTED": "0", "USE_NCCL": "0", "USE_CUDNN": "0",
                "USE_FLASH_ATTENTION": "0", "USE_MEM_EFF_ATTENTION": "0",
                "USE_CUSPARSELT": "0", "USE_CUFILE": "0", "USE_KINETO": "0",
                "USE_ITT": "0", "USE_MKLDNN": "0", "USE_NNPACK": "0",
                "BUILD_SOX": "0", "BUILD_KALDI": "0", "USE_FFMPEG": "0",
                "TORCHVISION_USE_NVJPEG": "0", "CMAKE_POLICY_VERSION_MINIMUM": "3.5"})
    if sys.platform == "win32":
        env["DISTUTILS_USE_SDK"] = "1"
    if spec["name"] == "kepler30":
        env.update({"USE_FBGEMM": "0", "USE_QNNPACK": "0", "USE_PYTORCH_QNNPACK": "0", "USE_XNNPACK": "0", "USE_NUMPY": "1"})
    if sys.platform != "win32":
        env["LD_LIBRARY_PATH"] = str(cuda / "lib64") + ":" + env.get("LD_LIBRARY_PATH", "")
        if shutil.which("ccache"):
            env["CMAKE_C_COMPILER_LAUNCHER"] = "ccache"
            env["CMAKE_CXX_COMPILER_LAUNCHER"] = "ccache"
            env["CMAKE_CUDA_COMPILER_LAUNCHER"] = "ccache"
    pyminor = sys.version_info.minor
    if pyminor <= 8:
        pip("install", "--prefer-binary", "setuptools<60", "wheel<0.38", "cmake<3.20", "ninja", "numpy<1.20", "pyyaml<6", "typing_extensions", "cffi", "future", "six", "requests", "pillow<10")
    else:
        # Newest sdists can drop an experimental ABI before older wheels do.
        # CFFI is only needed by the 1.6 lane and rejects CPython 3.13t entirely.
        # 2.7's requirements.txt explicitly caps setuptools on Windows.
        setuptools = "setuptools<=75.8.2" if spec["name"] == "kepler" else "setuptools<82"
        pip("install", "--prefer-binary", setuptools, "wheel", "cmake>=3.27,<4", "ninja", "numpy<3", "pyyaml", "typing_extensions", "packaging", "six", "requests", "pillow", "filelock", "sympy", "networkx", "jinja2", "fsspec", "psutil", "astunparse", "optree>=0.13.0")
    if sys.platform == "win32":
        pip("install", "delvewheel")
    else:
        # Ubuntu ships older patchelf than current auditwheel accepts.
        pip("install", "auditwheel", "patchelf>=0.14.5")
    sources = {}
    wheels = []
    torch_version = spec["torch"] + "+" + spec["local"]
    env["PYTORCH_BUILD_VERSION"] = torch_version
    env["PYTORCH_BUILD_NUMBER"] = "1"
    env["PYTORCH_VERSION"] = torch_version
    env["PYTORCH_VERSION_GE"] = torch_version
    # Modern torchvision supports third-party pins via a GE/LT pair.
    major, minor, _ = spec["torch"].split(".")
    env["PYTORCH_VERSION_LT"] = major + "." + str(int(minor) + 1)
    for repository, package, version in (("pytorch", "torch", spec["torch"]), ("vision", "torchvision", spec["vision"]), ("audio", "torchaudio", spec["audio"])):
        source = work / repository
        sources[repository] = checkout(repository, version, source)
        if package == "torch" and spec["name"] == "kepler30":
            patch_sm30(source)
        env["BUILD_VERSION"] = version + "+" + spec["local"]
        raw = work / (package + "-wheels")
        raw.mkdir(exist_ok=True)
        # New torch uses scikit-build-core; old versions use their original
        # setup.py. No upstream workflow or build implementation is vendored.
        project = source / "pyproject.toml"
        modern_backend = package == "torch" and project.exists() and "scikit_build_core.build" in project.read_text()
        if modern_backend:
            pip("wheel", "--no-deps", "--wheel-dir", raw, source, env=env)
        else:
            run(sys.executable, "setup.py", "bdist_wheel", "--dist-dir", raw, cwd=str(source), env=env)
        built = list(raw.glob("*.whl"))
        if len(built) != 1:
            raise RuntimeError("Expected exactly one upstream wheel for " + package)
        repaired = work / (package + "-repaired")
        repaired.mkdir(exist_ok=True)
        # auditwheel must see the toolkit and the already installed torch libs.
        os.environ.update({k: env[k] for k in ("LD_LIBRARY_PATH",) if k in env})
        repair(built[0], repaired, cuda)
        repaired_wheels = list(repaired.glob("*.whl"))
        if len(repaired_wheels) != 1:
            raise RuntimeError("Wheel repair produced an unexpected result")
        wheel = repaired_wheels[0]
        pip("install", "--no-deps", "--force-reinstall", wheel)
        shutil.copy2(str(wheel), str(out / wheel.name))
        wheels.append(out / wheel.name)
    # Resolve runtime dependencies without letting pip replace the local torch.
    constraints = work / "constraints.txt"
    constraints.write_text("torch==" + torch_version + "\n")
    pip("install", "--prefer-binary", "--constraint", constraints, *wheels)
    # Verify in a fresh environment, without the toolkit on the loader path.
    # Otherwise a broken wheel could pass by finding unbundled build libraries.
    fresh = work / "smoke-env"
    fresh_python = create_smoke_environment(fresh)
    run(fresh_python, "-m", "pip", "install", "--prefer-binary", "--constraint", constraints, *wheels)
    clean = env.copy()
    clean.pop("LD_LIBRARY_PATH", None)
    clean.pop("PYTHONPATH", None)
    clean["PATH"] = os.pathsep.join(p for p in clean.get("PATH", "").split(os.pathsep) if str(cuda).lower() not in p.lower())
    smoke = root / "scripts/smoke.py"
    run(fresh_python, smoke, "--spec", str(Path(args.spec).resolve()), "--report", str(out / "smoke.json"), cwd=str(work), env=clean)
    manifest = dict(spec, sources=sources, verification=json.loads((out / "smoke.json").read_text()),
                    wheels=[{"name": w.name, "sha256": hashlib.sha256(w.read_bytes()).hexdigest(), "size": w.stat().st_size} for w in wheels],
                    disabled_features=["cuDNN", "distributed/NCCL", "FlashAttention", "memory-efficient attention", "cuSPARSELt", "cuFile", "Kineto", "MKLDNN", "NNPACK", "FFmpeg", "SoX", "Kaldi", "NVJPEG"],
                    source_patch="cpp_extension sm_30 allowlist" if spec["name"] == "kepler30" else None)
    (out / "smoke.json").unlink()
    (out / (spec["id"] + ".json")).write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
