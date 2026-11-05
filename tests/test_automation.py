import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build import patch_sm30
import build
from plan import compatible_domain, matrix_rows, python_versions, release_tag, stable_releases
from publish import finalize, publish_bundle, validate_bundle


class PlanningTests(unittest.TestCase):
    def test_release_filter_and_semver_sort(self):
        releases = [{"tag_name": tag, "draft": draft, "prerelease": pre} for tag, draft, pre in
                    [("v2.9.0", False, False), ("v2.10.0", False, False), ("v2.11.0", True, False),
                     ("v2.12.0rc1", False, True), ("nightly", False, False)]]
        self.assertEqual(stable_releases(releases, "2.9.0"), ["2.9.0", "2.10.0"])

    def test_python_and_free_threaded(self):
        fixture = json.loads((ROOT / "tests/fixtures.json").read_text())
        self.assertEqual(python_versions(fixture["document"], "2.15.0"),
                         ["3.11", "3.12", "3.13", "3.14", "3.15", "3.13t", "3.14t", "3.15t"])

    def test_legacy_python_row(self):
        self.assertEqual(python_versions("| 2.7 | >=3.9, <=3.13, (3.13t experimental) | C++17 |", "2.7.1"),
                         ["3.9", "3.10", "3.11", "3.12", "3.13", "3.13t"])

    def test_unknown_python_fails_closed(self):
        with self.assertRaises(ValueError):
            python_versions("", "2.16.0")
        with self.assertRaises(ValueError):
            python_versions("| 2.16 | unknown |", "2.16.0")

    def test_domain_actual_requirements_not_minor_arithmetic(self):
        self.assertTrue(compatible_domain({"requires_dist": ["torch>=2.14", "numpy"]}, "2.15.0"))
        self.assertFalse(compatible_domain({"requires_dist": ["torch==2.7.1"]}, "2.15.0"))
        self.assertTrue(compatible_domain({"requires_dist": ["torch==2.7.1"]}, "2.7.1"))
        self.assertFalse(compatible_domain({}, "2.15.0"))
        self.assertFalse(compatible_domain({"requires_dist": ["torch"]}, "1.6.0"))

    def rows(self):
        fixture = json.loads((ROOT / "tests/fixtures.json").read_text())
        config = json.loads((ROOT / "config.json").read_text())
        domains = {(name, v): fixture[name] for name in ("torchvision", "torchaudio") for v in ("1.6.0", "2.7.1", "2.15.0")}
        return matrix_rows(config, fixture["versions"], {"2.15.0": fixture["document"]}, domains)

    def test_unique_jobs_and_versions(self):
        rows = self.rows()
        self.assertLessEqual(len(rows), 256)
        self.assertEqual(len({r["id"] for r in rows}), len(rows))
        for row in rows:
            self.assertEqual(len(row["fingerprint"]), 64)
            self.assertIn(row["python"].replace(".", ""), row["local"])

    def test_old_lanes_are_not_rebuilt_as_latest(self):
        for row in self.rows():
            if row["name"] == "kepler":
                self.assertEqual(row["torch"], "2.7.1")
                self.assertEqual(row["architectures"], ["3.5", "3.7"])
            if row["name"] == "kepler30":
                self.assertEqual(row["torch"], "1.6.0")

    def test_arm_is_native_and_not_jetson(self):
        for row in self.rows():
            if row["platform"] == "linux-aarch64":
                self.assertEqual(row["runner"], "ubuntu-24.04-arm")
                self.assertNotIn(row["name"], ("kepler", "kepler30", "maxwell", "pascal"))
                self.assertNotIn("5.3", row["architectures"])

    def test_revision_separates_immutable_releases(self):
        self.assertNotEqual(release_tag("2.15.0", "maxwell", 1), release_tag("2.15.0", "maxwell", 2))


class PublicationTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / ".tmp"
        base.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.spec = {"id": "test-bundle", "fingerprint": "abc", "tag": "test-tag"}
        wheels = []
        for package in ("torch", "torchvision", "torchaudio"):
            path = self.directory / (package + "-1.0-py3-none-any.whl")
            path.write_bytes(package.encode())
            wheels.append({"name": path.name, "size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        self.manifest = dict(self.spec, wheels=wheels, verification={"imports": True, "cpu_ops": True})
        self.save()

    def save(self):
        (self.directory / "test-bundle.json").write_text(json.dumps(self.manifest))

    def test_complete_bundle(self):
        files, manifest = validate_bundle(self.spec, self.directory)
        self.assertEqual(len(files), 3)
        self.assertEqual(manifest.name, "test-bundle.json")

    def test_missing_domain_prevents_publication(self):
        self.manifest["wheels"].pop()
        self.save()
        with self.assertRaises(ValueError):
            validate_bundle(self.spec, self.directory)

    def test_tampering_prevents_publication(self):
        (self.directory / self.manifest["wheels"][0]["name"]).write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            validate_bundle(self.spec, self.directory)

    def test_bad_smoke_prevents_publication(self):
        self.manifest["verification"]["imports"] = False
        self.save()
        with self.assertRaises(ValueError):
            validate_bundle(self.spec, self.directory)

    def test_wrong_fingerprint_prevents_publication(self):
        self.manifest["fingerprint"] = "different"
        self.save()
        with self.assertRaises(ValueError):
            validate_bundle(self.spec, self.directory)

    def test_manifest_is_uploaded_last(self):
        gh = Mock(repository="test/repo")
        gh.release.return_value = {"id": 1, "draft": True}
        gh.assets.return_value = []
        publish_bundle(gh, self.spec, self.directory)
        self.assertEqual(gh.upload.call_args_list[-1].args[1].name, "test-bundle.json")

    def test_completed_bundle_is_not_overwritten(self):
        gh = Mock(repository="test/repo")
        gh.release.return_value = {"id": 1, "draft": False}
        gh.assets.return_value = [{"name": "test-bundle.json"}]
        publish_bundle(gh, self.spec, self.directory)
        gh.upload.assert_not_called()

    def test_incomplete_release_stays_draft(self):
        gh = Mock(repository="test/repo")
        gh.release.return_value = {"id": 1, "draft": True}
        gh.assets.return_value = []
        finalize(gh, [self.spec])
        gh.request.assert_not_called()

    def test_complete_release_is_published(self):
        gh = Mock(repository="test/repo")
        gh.release.return_value = {"id": 1, "draft": True}
        gh.assets.return_value = [{"name": "test-bundle.json"}]
        finalize(gh, [self.spec])
        self.assertEqual(gh.request.call_args.args[1], "PATCH")
        self.assertFalse(gh.request.call_args.args[2]["draft"])


class BuildCompatibilityTests(unittest.TestCase):
    def test_fresh_smoke_environment_upgrades_pip_before_wheel_installation(self):
        with patch("build.sys.platform", "linux"), patch("build.run") as run:
            python = build.create_smoke_environment(Path("smoke-env"))
        self.assertEqual(python, Path("smoke-env/bin/python"))
        self.assertEqual(run.call_args_list[0].args, (sys.executable, "-m", "venv", Path("smoke-env")))
        self.assertEqual(run.call_args_list[1].args, (python, "-m", "pip", "install", "--upgrade", "pip"))

    def test_linux_repair_explicitly_targets_build_host_glibc(self):
        for version, machine in (("2.27", "x86_64"), ("2.35", "x86_64"), ("2.39", "aarch64")):
            with self.subTest(version=version, machine=machine), patch("build.sys.platform", "linux"), patch("build.os.confstr", return_value="glibc " + version), patch("platform.libc_ver", return_value=("glibc", "2.17")), patch("platform.machine", return_value=machine), patch("build.run") as run:
                build.repair(Path("torch-test.whl"), Path("out"), Path("cuda"))
                args = run.call_args.args
                self.assertIn("--plat", args)
                self.assertEqual(args[args.index("--plat") + 1], "manylinux_" + version.replace(".", "_") + "_" + machine)
                self.assertIn("libcuda.so.1", args)

    def test_linux_repair_does_not_guess_when_libc_is_unknown(self):
        with patch("build.sys.platform", "linux"), patch("build.os.confstr", return_value=""), patch("build.run") as run:
            with self.assertRaisesRegex(RuntimeError, "glibc"):
                build.repair(Path("torch-test.whl"), Path("out"), Path("cuda"))
            run.assert_not_called()

    def test_windows_repair_does_not_use_a_manylinux_tag(self):
        with patch("build.sys.platform", "win32"), patch("build.os.confstr") as libc, patch("build.run") as run:
            build.repair(Path("torch-test.whl"), Path("out"), Path("cuda"))
            self.assertIn("delvewheel", run.call_args.args)
            self.assertNotIn("--plat", run.call_args.args)
            libc.assert_not_called()

    def test_checkout_enables_long_paths_for_clone_and_recursive_submodules(self):
        with patch("build.run") as run, patch("build.subprocess.check_output", return_value=b"sha\n"):
            build.checkout("pytorch", "2.7.1", Path("source"))
        for call in run.call_args_list:
            self.assertEqual(call.args[:3], ("git", "-c", "core.longpaths=true"))

    def bootstrap_command(self, minor, index=0):
        (ROOT / ".tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
            root = Path(directory)
            spec = root / "spec.json"
            spec.write_text(json.dumps({"name": "kepler", "architectures": ["3.5", "3.7"],
                                        "torch": "2.7.1", "vision": "0.22.1", "audio": "2.7.1", "local": "test"}))
            with patch("build.Path.cwd", return_value=root), patch("build.sys.argv", ["build.py", "--spec", str(spec), "--output", str(root / "dist")]), patch("build.sys.version_info", Mock(minor=minor)), patch("build.pip") as pip, patch("build.checkout", side_effect=RuntimeError("stop before compilation")), patch.dict(os.environ):
                with self.assertRaisesRegex(RuntimeError, "stop before compilation"):
                    build.main()
            return pip.call_args_list[index].args

    def test_linux_repair_installs_patchelf_new_enough_for_auditwheel(self):
        with patch("build.sys.platform", "linux"):
            self.assertIn("patchelf>=0.14.5", self.bootstrap_command(7, 1))

    def test_modern_bootstrap_does_not_require_cffi_on_python313t(self):
        self.assertNotIn("cffi", self.bootstrap_command(13))

    def test_bootstrap_prefers_compatible_wheels_over_newer_sdists(self):
        self.assertIn("--prefer-binary", self.bootstrap_command(13))

    def test_kepler_setuptools_respects_upstream_windows_limit(self):
        self.assertIn("setuptools<=75.8.2", self.bootstrap_command(13))

    def test_cuda102_installer_has_libxml2_before_it_runs(self):
        script = (ROOT / "scripts/linux.sh").read_text()
        apt = script.split("apt-get install", 1)[1].split("git config", 1)[0]
        self.assertIn("libxml2", apt.split())

    def test_conda_python_is_only_a_base_for_an_isolated_build_venv(self):
        script = (ROOT / "scripts/linux.sh").read_text()
        self.assertIn('-p /opt/python-base "python=$PYTHON_VERSION"', script)
        self.assertIn('/opt/python-base/bin/python -m venv /opt/python', script)
        self.assertNotIn('--system-site-packages', script)

    def test_build_scripts_parse_on_python36(self):
        for name in ("build.py", "smoke.py"):
            ast.parse((ROOT / "scripts" / name).read_text(), feature_version=(3, 6))

    def test_sm30_patch_is_narrow_and_fails_closed(self):
        base = ROOT / ".tmp"
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            source = Path(directory)
            target = source / "torch/utils/cpp_extension.py"
            target.parent.mkdir(parents=True)
            target.write_text("supported_arches = ['3.5', '3.7']\n")
            patch_sm30(source)
            self.assertEqual(target.read_text(), "supported_arches = ['3.0', '3.5', '3.7']\n")
            with self.assertRaises(RuntimeError):
                patch_sm30(source)


if __name__ == "__main__":
    unittest.main()
