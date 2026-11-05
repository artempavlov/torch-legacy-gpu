"""Draft-first publication. A bundle's manifest is its final commit marker."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.error

from github_api import GitHub


def validate_bundle(spec, directory):
    path = directory / (spec["id"] + ".json")
    manifest = json.loads(path.read_text())
    if manifest["fingerprint"] != spec["fingerprint"]:
        raise ValueError("Manifest does not match this build plan")
    wheels = manifest["wheels"]
    if len(wheels) != 3 or {w["name"].split("-")[0] for w in wheels} != {"torch", "torchvision", "torchaudio"}:
        raise ValueError("All three wheels are required")
    files = []
    for wheel in wheels:
        if Path(wheel["name"]).name != wheel["name"]:
            raise ValueError("Unsafe asset name")
        asset = directory / wheel["name"]
        if asset.stat().st_size != wheel["size"] or hashlib.sha256(asset.read_bytes()).hexdigest() != wheel["sha256"]:
            raise ValueError("Wheel hash/size mismatch: " + asset.name)
        files.append(asset)
    verification = manifest["verification"]
    if not verification["imports"] or not verification["cpu_ops"]:
        raise ValueError("Smoke checks did not pass")
    return files, path


def ensure_release(gh, spec):
    release = gh.release(spec["tag"])
    if release:
        return release
    try:
        return gh.request(f"/repos/{gh.repository}/releases", "POST", {
            "tag_name": spec["tag"], "target_commitish": "main", "draft": True,
            "name": "PyTorch " + spec["torch"] + " / " + spec["name"],
            "body": "Unofficial source builds. See README for installation, disabled features and platform limits.\n\n"
                    "Each bundle has torch, torchvision, torchaudio and a JSON manifest with source commits, SHA-256 hashes and verification results. "
                    "GPU execution is not tested on hosted runners. Incomplete groups remain drafts.\n"})
    except urllib.error.HTTPError as error:
        if error.code != 422:
            raise
        release = gh.release(spec["tag"])
        if not release:
            raise
        return release


def publish_bundle(gh, spec, directory):
    files, manifest = validate_bundle(spec, directory)
    release = ensure_release(gh, spec)
    existing = {a["name"]: a for a in gh.assets(release)}
    if manifest.name in existing:
        return
    if not release["draft"]:
        raise ValueError("Refusing to change an already published release; bump revision")
    # A previous interrupted attempt may have left some wheels but no manifest.
    # Only this exact ABI/platform bundle's asset names may be replaced.
    for asset in files + [manifest]:
        if asset.name in existing:
            gh.request(f"/repos/{gh.repository}/releases/assets/{existing[asset.name]['id']}", "DELETE")
        gh.upload(release, asset)


def finalize(gh, rows):
    groups = {}
    for spec in rows:
        groups.setdefault(spec["tag"], []).append(spec)
    for tag, specs in groups.items():
        release = gh.release(tag)
        if not release or not release["draft"]:
            continue
        names = {a["name"] for a in gh.assets(release)}
        required = {s["id"] + ".json" for s in specs}
        if required <= names:
            gh.request(f"/repos/{gh.repository}/releases/{release['id']}", "PATCH", {"draft": False, "make_latest": "false"})
            print("Published", tag)
        else:
            print("Still draft:", tag, "missing", len(required - names), "bundles")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec")
    parser.add_argument("--directory", default="dist")
    parser.add_argument("--finalize", help="Full catalog JSON, not the pending matrix")
    args = parser.parse_args()
    gh = GitHub()
    if args.finalize:
        finalize(gh, json.loads(Path(args.finalize).read_text())["include"])
    else:
        publish_bundle(gh, json.loads(Path(args.spec).read_text()), Path(args.directory))


if __name__ == "__main__":
    main()
