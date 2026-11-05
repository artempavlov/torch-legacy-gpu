"""Resolve real upstream versions and plan only unfinished wheel bundles."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.request

from packaging.requirements import Requirement
from packaging.version import Version

from github_api import GitHub


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "torch-legacy-gpu"}), timeout=60) as response:
        return response.read().decode()


def stable_releases(releases, minimum):
    return sorted({r["tag_name"].removeprefix("v") for r in releases
                   if not r["draft"] and not r["prerelease"]
                   and re.fullmatch(r"v\d+\.\d+\.\d+", r["tag_name"])
                   and Version(r["tag_name"][1:]) >= Version(minimum)}, key=Version)


def python_versions(document, version):
    minor = ".".join(version.split(".")[:2])
    row = next((line for line in document.splitlines() if re.match(rf"\|\s*{re.escape(minor)}\s*\|", line)), None)
    if row is None:
        raise ValueError(f"No upstream Python compatibility row for {version}")
    cell = row.split("|")[2]
    bounds = re.findall(r"(?:>=|<=)\s*\(?\s*3\.(\d+)", cell)
    if len(bounds) != 2:
        raise ValueError(f"Cannot safely parse Python compatibility: {cell}")
    low, high = map(int, bounds)
    versions = [f"3.{i}" for i in range(low, high + 1)]
    if re.search(r"3\.\d+t", cell):
        versions += [f"3.{i}t" for i in range(max(13, low), high + 1)]
    return versions


def compatible_domain(metadata, torch_version):
    requirements = [Requirement(r) for r in metadata.get("requires_dist") or []]
    torch = [r for r in requirements if r.name.lower() == "torch"]
    # An unversioned "torch" requirement is not compatibility evidence.
    return bool(torch) and all(str(r.specifier) and Version(torch_version) in r.specifier for r in torch)


def domain_version(package, torch_version, cache):
    key = (package, torch_version)
    if key in cache:
        return cache[key]
    versions = json.loads(fetch(f"https://pypi.org/pypi/{package}/json"))["releases"]
    candidates = sorted((v for v, files in versions.items() if files and not Version(v).is_prerelease
                         and not Version(v).is_devrelease and not all(f.get("yanked") for f in files)), key=Version, reverse=True)
    for version in candidates:
        info = json.loads(fetch(f"https://pypi.org/pypi/{package}/{version}/json"))["info"]
        if compatible_domain(info, torch_version):
            cache[key] = version
            return version
    raise ValueError(f"No {package} release declares compatibility with torch {torch_version}")


def release_tag(version, lane, revision):
    return f"torch-v{version}-{lane}-r{revision}"


def matrix_rows(config, releases, documents, domains):
    rows = []
    for lane in config["lanes"]:
        versions = releases if lane["torch"] == "latest" else [lane["torch"]]
        for version in versions:
            pythons = lane.get("python") or python_versions(documents[version], version)
            for platform in lane["platforms"]:
                arches = lane.get("arm_architectures", lane["architectures"]) if platform == "linux-aarch64" else lane["architectures"]
                for python in pythons:
                    abi = "cp" + python.replace(".", "")
                    local = "cu" + "".join(lane["cuda"].split(".")[:2]) + "." + lane["name"] + "." + abi
                    row = dict(lane, torch=version, python=python, platform=platform, architectures=arches,
                               vision=domains[("torchvision", version)], audio=domains[("torchaudio", version)],
                               tag=release_tag(version, lane["name"], config["revision"]), local=local,
                               runner={"linux-x86_64": "ubuntu-22.04", "linux-aarch64": "ubuntu-24.04-arm", "windows-x86_64": "windows-2022"}[platform])
                    row["id"] = f"{lane['name']}-{version}-{platform}-{abi}-r{config['revision']}"
                    row["fingerprint"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                    rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch", default="")
    parser.add_argument("--only", default="", help="Exact bundle ID, or empty for all")
    parser.add_argument("--offline", action="store_true", help="Use checked-in representative test fixtures")
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", ""))
    args = parser.parse_args()
    config = json.loads(Path("config.json").read_text())
    if not args.offline and not args.repository:
        parser.error("--repository OWNER/REPO is required outside GitHub Actions")
    gh = GitHub(args.repository)
    if args.offline:
        fixture = json.loads(Path("tests/fixtures.json").read_text())
        versions = fixture["versions"]
        documents = {v: fixture["document"] for v in versions}
        domains = {(name, v): fixture[name] for v in versions + ["1.6.0", "2.7.1"] for name in ("torchvision", "torchaudio")}
    else:
        if args.torch:
            if not re.fullmatch(r"\d+\.\d+\.\d+", args.torch):
                raise ValueError("--torch must be a stable X.Y.Z version")
            release = gh.request(f"/repos/pytorch/pytorch/releases/tags/v{args.torch}")
            versions = stable_releases([release], args.torch)
        else:
            releases = gh.request("/repos/pytorch/pytorch/releases?per_page=100")
            versions = stable_releases(releases, config["first_release"])
        if not versions:
            raise ValueError("No eligible upstream release found")
        documents = {v: fetch(f"https://raw.githubusercontent.com/pytorch/pytorch/v{v}/RELEASE.md") for v in versions}
        domains = {}
        pinned = {(package, lane["torch"]): lane[field]
                  for lane in config["lanes"] for package, field in (("torchvision", "vision"), ("torchaudio", "audio"))
                  if field in lane}
        for version in sorted(set(versions + [l["torch"] for l in config["lanes"] if l["torch"] != "latest"]), key=Version):
            for package in ("torchvision", "torchaudio"):
                if (package, version) in pinned:
                    domains[(package, version)] = pinned[(package, version)]
                else:
                    domain_version(package, version, domains)
    rows = matrix_rows(config, versions, documents, domains)
    Path("catalog.json").write_text(json.dumps({"include": rows}, indent=2) + "\n")
    if args.only:
        rows = [r for r in rows if r["id"] == args.only]
        if not rows:
            raise ValueError("Unknown bundle ID")
    # Completed bundles are immutable. A revision bump is required to rebuild.
    completed = {}
    pending = []
    for row in rows:
        if not args.offline and row["tag"] not in completed:
            release = gh.release(row["tag"])
            completed[row["tag"]] = {a["name"] for a in gh.assets(release)} if release else set()
        if row["id"] + ".json" not in completed.get(row["tag"], set()):
            pending.append(row)
    if len(pending) > 256:
        # Drain a backlog over subsequent runs, never exceed the Actions limit.
        pending = pending[:256]
    Path("plan.json").write_text(json.dumps({"include": pending}, indent=2) + "\n")
    print(json.dumps({"include": pending}))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write("matrix=" + json.dumps({"include": pending}, separators=(",", ":")) + "\n")
            output.write(f"count={len(pending)}\n")


if __name__ == "__main__":
    main()
