# SPDX-License-Identifier: GPL-3.0-only
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from hermeto.core.constants import Mode
from hermeto.core.errors import PackageRejected
from hermeto.core.models.input import CargoPackageInput, CargoPackageSelection, Request
from hermeto.core.package_managers.cargo.main import (
    CargoVendorResult,
    _cargo_vendor_filterer_cmd,
    _contains_stub_crates,
    _fetch_dependencies,
    _generate_sbom_components,
    _is_stub_crate,
    _merge_vendored_crates,
    _read_crate_ids,
)
from hermeto.core.rooted_path import RootedPath

CONFIG_TEMPLATE = '[source.vendored-sources]\ndirectory = "/somewhere/vendor"\n'


def _checksums(files: dict[str, bytes]) -> str:
    return json.dumps(
        {
            "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
            "package": "0" * 64,
        }
    )


def make_real_crate(vendor_dir: Path, name: str, version: str) -> Path:
    """Write a crate directory the way `cargo vendor --versioned-dirs` lays it out."""
    crate = vendor_dir / f"{name}-{version}"
    files = {
        "Cargo.toml": f'[package]\nname = "{name}"\nversion = "{version}"\n'.encode(),
        "Cargo.toml.orig": b"",
        "src/lib.rs": b"pub fn real() {}\n",
    }
    for rel, data in files.items():
        (crate / rel).parent.mkdir(parents=True, exist_ok=True)
        (crate / rel).write_bytes(data)
    (crate / ".cargo-checksum.json").write_text(_checksums(files))
    return crate


def make_stub_crate(vendor_dir: Path, name: str, version: str) -> Path:
    """Write a crate directory the way cargo-vendor-filterer stubs a filtered-out crate."""
    crate = vendor_dir / f"{name}-{version}"
    files = {
        "Cargo.toml": f'[package]\nname = "{name}"\nversion = "{version}"\n'.encode(),
        "src/lib.rs": b"",
    }
    for rel, data in files.items():
        (crate / rel).parent.mkdir(parents=True, exist_ok=True)
        (crate / rel).write_bytes(data)
    (crate / ".cargo-checksum.json").write_text(_checksums(files))
    return crate


def _is_real(crate: Path) -> bool:
    return (crate / "src" / "lib.rs").read_text() != ""


class TestStubDetection:
    def test_real_crate_is_not_a_stub(self, tmp_path: Path) -> None:
        assert not _is_stub_crate(make_real_crate(tmp_path, "real", "1.0.0"))

    def test_filterer_stub_is_a_stub(self, tmp_path: Path) -> None:
        assert _is_stub_crate(make_stub_crate(tmp_path, "stub", "1.0.0"))

    def test_crate_without_checksum_file_is_not_a_stub(self, tmp_path: Path) -> None:
        crate = make_stub_crate(tmp_path, "stub", "1.0.0")
        (crate / ".cargo-checksum.json").unlink()
        assert not _is_stub_crate(crate)

    def test_crate_with_corrupt_checksum_file_is_not_a_stub(self, tmp_path: Path) -> None:
        crate = make_stub_crate(tmp_path, "stub", "1.0.0")
        (crate / ".cargo-checksum.json").write_text("{not json")
        assert not _is_stub_crate(crate)

    def test_crate_with_non_empty_lib_is_not_a_stub(self, tmp_path: Path) -> None:
        crate = make_stub_crate(tmp_path, "stub", "1.0.0")
        (crate / "src" / "lib.rs").write_text("pub fn f() {}\n")
        assert not _is_stub_crate(crate)

    def test_vendor_dir_with_a_stub_contains_stubs(self, tmp_path: Path) -> None:
        make_real_crate(tmp_path, "real", "1.0.0")
        assert not _contains_stub_crates(tmp_path)
        make_stub_crate(tmp_path, "stub", "1.0.0")
        assert _contains_stub_crates(tmp_path)

    def test_missing_vendor_dir_contains_no_stubs(self, tmp_path: Path) -> None:
        assert not _contains_stub_crates(tmp_path / "missing")

    def test_read_crate_ids(self, tmp_path: Path) -> None:
        make_real_crate(tmp_path, "real", "1.0.0")
        make_stub_crate(tmp_path, "stub", "2.0.0")
        assert _read_crate_ids(tmp_path, real_only=True) == {("real", "1.0.0")}
        assert _read_crate_ids(tmp_path, real_only=False) == {
            ("real", "1.0.0"),
            ("stub", "2.0.0"),
        }


class TestMergeVendoredCrates:
    def test_new_crates_are_moved(self, tmp_path: Path) -> None:
        staging, vendor = tmp_path / "staging", tmp_path / "vendor"
        make_real_crate(staging, "real", "1.0.0")
        make_stub_crate(staging, "stub", "1.0.0")

        _merge_vendored_crates(staging, vendor)

        assert _is_real(vendor / "real-1.0.0")
        assert _is_stub_crate(vendor / "stub-1.0.0")
        assert not any(staging.iterdir())

    def test_real_crate_replaces_existing_stub(self, tmp_path: Path) -> None:
        staging, vendor = tmp_path / "staging", tmp_path / "vendor"
        make_stub_crate(vendor, "shared", "1.0.0")
        make_real_crate(staging, "shared", "1.0.0")

        _merge_vendored_crates(staging, vendor)

        assert _is_real(vendor / "shared-1.0.0")

    def test_stub_never_replaces_existing_real_crate(self, tmp_path: Path) -> None:
        staging, vendor = tmp_path / "staging", tmp_path / "vendor"
        make_real_crate(vendor, "shared", "1.0.0")
        make_stub_crate(staging, "shared", "1.0.0")

        _merge_vendored_crates(staging, vendor)

        assert _is_real(vendor / "shared-1.0.0")

    def test_existing_real_crate_is_kept(self, tmp_path: Path) -> None:
        staging, vendor = tmp_path / "staging", tmp_path / "vendor"
        existing = make_real_crate(vendor, "shared", "1.0.0")
        (existing / "marker").write_text("")
        make_real_crate(staging, "shared", "1.0.0")

        _merge_vendored_crates(staging, vendor)

        assert (vendor / "shared-1.0.0" / "marker").exists()


def _selection(**kwargs: Any) -> CargoPackageSelection:
    return CargoPackageSelection(**kwargs)


class TestCargoVendorFiltererCmd:
    def test_packages_features_and_platforms(self) -> None:
        package = CargoPackageInput(
            type="cargo",
            packages=[
                _selection(name="server", no_default_features=True, features=["openssl"]),
                _selection(name="operator", no_default_features=True, features=["openssl", "x"]),
            ],
            platforms=["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"],
        )
        assert package.packages is not None

        cmd = _cargo_vendor_filterer_cmd(package.packages, package, Path("/out/vendor"))

        assert cmd == [
            "cargo-vendor-filterer",
            "--locked",
            "--versioned-dirs",
            "--respect-source-config",
            "--keep-dep-kinds",
            "no-dev",
            "--package",
            "server",
            "--package",
            "operator",
            "--features",
            "server/openssl,operator/openssl,operator/x",
            "--no-default-features",
            "--platform",
            "x86_64-unknown-linux-gnu",
            "--platform",
            "aarch64-unknown-linux-gnu",
            "/out/vendor",
        ]

    def test_all_features_without_platforms(self) -> None:
        package = CargoPackageInput(
            type="cargo", packages=[_selection(name="a", all_features=True)]
        )
        assert package.packages is not None

        cmd = _cargo_vendor_filterer_cmd(package.packages, package, Path("/out/vendor"))

        assert cmd[-3:] == ["a", "--all-features", "/out/vendor"]
        assert "--platform" not in cmd
        assert "--features" not in cmd

    @pytest.mark.parametrize(
        "packages",
        [
            pytest.param(
                [{"name": "a", "no_default_features": True}, {"name": "b"}],
                id="mixed_no_default_features",
            ),
            pytest.param(
                [{"name": "a", "all_features": True}, {"name": "b"}],
                id="mixed_all_features",
            ),
        ],
    )
    def test_mixed_default_feature_flags_are_rejected(self, packages: list[dict[str, Any]]) -> None:
        package = CargoPackageInput.model_validate({"type": "cargo", "packages": packages})
        assert package.packages is not None

        with pytest.raises(PackageRejected, match="must agree on 'no_default_features'"):
            _cargo_vendor_filterer_cmd(package.packages, package, Path("/out/vendor"))


def _request(source_dir: Path, output_dir: Path, package: dict[str, Any]) -> Request:
    output_dir.mkdir(parents=True, exist_ok=True)
    return Request(source_dir=source_dir, output_dir=output_dir, packages=[package])


def _vendor_into_last_arg(*crates: tuple[str, str, bool]) -> Any:
    """Stand in for the vendoring tool: create the given crates in the directory it is given."""

    def run(cmd: list[str], params: dict[str, Any], package_dir: Path) -> CargoVendorResult:
        target = Path(cmd[-1])
        for name, version, real in crates:
            (make_real_crate if real else make_stub_crate)(target, name, version)
        return CargoVendorResult(config_template=CONFIG_TEMPLATE, lockfile_was_generated=False)

    return run


RUN_CMD = "hermeto.core.package_managers.cargo.main._run_cmd_watching_out_for_lock_mismatch"


class TestFetchDependencies:
    @mock.patch(RUN_CMD)
    def test_unfiltered_input_vendors_in_place(
        self, mock_run: mock.Mock, rooted_tmp_path: RootedPath
    ) -> None:
        request = _request(rooted_tmp_path.path, rooted_tmp_path.path / "out", {"type": "cargo"})
        mock_run.return_value = CargoVendorResult(CONFIG_TEMPLATE, False)

        result = _fetch_dependencies(rooted_tmp_path, request, request.cargo_packages[0])

        assert mock_run.call_args.kwargs["cmd"] == [
            "cargo",
            "vendor",
            "--locked",
            "--versioned-dirs",
            "--no-delete",
            "--respect-source-config",
            str(rooted_tmp_path.path / "out" / "deps" / "cargo"),
        ]
        assert result.reachable_crates is None

    @mock.patch(RUN_CMD)
    def test_filtered_input_reports_reachable_crates(
        self, mock_run: mock.Mock, rooted_tmp_path: RootedPath
    ) -> None:
        output_dir = rooted_tmp_path.path / "out"
        request = _request(
            rooted_tmp_path.path,
            output_dir,
            {"type": "cargo", "packages": [{"name": "server", "features": ["openssl"]}]},
        )
        mock_run.side_effect = _vendor_into_last_arg(
            ("openssl", "0.10.0", True), ("ring", "0.17.0", False)
        )

        result = _fetch_dependencies(rooted_tmp_path, request, request.cargo_packages[0])

        cmd = mock_run.call_args.kwargs["cmd"]
        assert cmd[0] == "cargo-vendor-filterer"
        assert "--no-delete" not in cmd
        assert result.reachable_crates == {("openssl", "0.10.0")}
        assert result.config_template == CONFIG_TEMPLATE
        vendor = output_dir / "deps" / "cargo"
        assert _is_real(vendor / "openssl-0.10.0")
        assert _is_stub_crate(vendor / "ring-0.17.0")
        assert [p.name for p in (output_dir / "deps").iterdir()] == ["cargo"]

    @mock.patch(RUN_CMD)
    def test_unfiltered_input_after_filtered_one_restores_stubbed_crates(
        self, mock_run: mock.Mock, rooted_tmp_path: RootedPath
    ) -> None:
        output_dir = rooted_tmp_path.path / "out"
        request = _request(rooted_tmp_path.path, output_dir, {"type": "cargo"})
        vendor = output_dir / "deps" / "cargo"
        make_stub_crate(vendor, "ring", "0.17.0")
        mock_run.side_effect = _vendor_into_last_arg(("ring", "0.17.0", True))

        result = _fetch_dependencies(rooted_tmp_path, request, request.cargo_packages[0])

        cmd = mock_run.call_args.kwargs["cmd"]
        assert cmd[:2] == ["cargo", "vendor"]
        assert "--no-delete" not in cmd
        assert result.reachable_crates is None
        assert _is_real(vendor / "ring-0.17.0")

    @mock.patch(RUN_CMD)
    def test_staging_dir_is_removed_when_vendoring_fails(
        self, mock_run: mock.Mock, rooted_tmp_path: RootedPath
    ) -> None:
        output_dir = rooted_tmp_path.path / "out"
        request = _request(
            rooted_tmp_path.path, output_dir, {"type": "cargo", "packages": [{"name": "server"}]}
        )
        mock_run.side_effect = PackageRejected("boom", solution=None)

        with pytest.raises(PackageRejected):
            _fetch_dependencies(rooted_tmp_path, request, request.cargo_packages[0])

        assert not any((output_dir / "deps").iterdir())


_CARGO_TOML = '[package]\nname = "my-crate"\nversion = "0.1.0"\n'
_CARGO_LOCK = """version = 3

[[package]]
name = "my-crate"
version = "0.1.0"

[[package]]
name = "openssl"
version = "0.10.0"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "aaaa"

[[package]]
name = "ring"
version = "0.17.0"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "bbbb"

[[package]]
name = "workspace-member"
version = "0.1.0"
"""


@pytest.mark.parametrize(
    "reachable, expected",
    [
        pytest.param(None, ["my-crate", "openssl", "ring", "workspace-member"], id="unfiltered"),
        pytest.param(
            frozenset({("openssl", "0.10.0")}),
            ["my-crate", "openssl", "workspace-member"],
            id="filtered_keeps_workspace_members",
        ),
    ],
)
@mock.patch("hermeto.core.package_managers.cargo.main.get_config")
@mock.patch("hermeto.core.package_managers.cargo.main.get_repo_id")
def test_generate_sbom_components_skips_unreachable_crates(
    mock_get_repo_id: mock.Mock,
    mock_get_config: mock.Mock,
    reachable: frozenset[tuple[str, str]] | None,
    expected: list[str],
    rooted_tmp_path: RootedPath,
) -> None:
    mock_get_config.return_value.mode = Mode.STRICT
    mock_get_config.return_value.cargo.proxy_url = None
    mock_get_repo_id.return_value.as_vcs_url_qualifier.return_value = "git+https://x@0"
    (rooted_tmp_path.path / "Cargo.toml").write_text(_CARGO_TOML)
    (rooted_tmp_path.path / "Cargo.lock").write_text(_CARGO_LOCK)
    request = _request(rooted_tmp_path.path, rooted_tmp_path.path.parent / "out", {"type": "cargo"})

    components = _generate_sbom_components(rooted_tmp_path, request, False, reachable)

    assert [c.name for c in components] == expected
