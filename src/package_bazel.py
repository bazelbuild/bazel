#!/usr/bin/env python3

import shutil
import stat
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)


def _safe_path(destination, name):
    target = (destination / name).resolve()
    if target != destination and destination not in target.parents:
        raise ValueError("archive entry escapes destination: {}".format(name))
    return target


def _extract_zip(archive, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            target = _safe_path(destination, info.filename)
            mode = info.external_attr >> 16
            if info.is_dir():
                target.mkdir(parents = True, exist_ok = True)
            elif stat.S_ISLNK(mode):
                target.parent.mkdir(parents = True, exist_ok = True)
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(source.read(info).decode("utf-8"))
            else:
                target.parent.mkdir(parents = True, exist_ok = True)
                target.write_bytes(source.read(info))
                if mode:
                    target.chmod(mode)


def _extract_tar(archive, destination):
    destination = destination.resolve()
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            target = _safe_path(destination, member.name)
            if member.isdir():
                target.mkdir(parents = True, exist_ok = True)
            elif member.isfile():
                target.parent.mkdir(parents = True, exist_ok = True)
                extracted = source.extractfile(member)
                if extracted is None:
                    raise ValueError("could not extract tar entry: {}".format(member.name))
                target.write_bytes(extracted.read())
                target.chmod(member.mode)
            else:
                raise ValueError("unsupported tar entry: {}".format(member.name))


def _zip_info(path, archive_name, compression):
    info = zipfile.ZipInfo(archive_name, FIXED_DATE_TIME)
    info.compress_type = compression
    info.create_system = 3
    info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
    return info


def _recompress_deploy_jar(source, output, workdir):
    _extract_zip(source, workdir)
    files = sorted(path for path in workdir.rglob("*") if path.is_file() and not path.is_symlink())
    with zipfile.ZipFile(output, "w", allowZip64 = False) as archive:
        for path in files:
            archive.writestr(
                _zip_info(path, path.relative_to(workdir).as_posix(), zipfile.ZIP_STORED),
                path.read_bytes(),
            )
    build_data = workdir / "build-data.properties"
    if build_data.exists():
        for line in build_data.read_text(encoding = "utf-8", errors = "replace").splitlines():
            if line.startswith("build.label="):
                return line.partition("=")[2] or "no_version"
    return "no_version"


def main():
    output = Path(sys.argv[1]).resolve()
    embedded_tools_arg = sys.argv[2]
    deploy_jar = Path(sys.argv[3]).resolve()
    install_base_key = Path(sys.argv[4]).resolve()
    platforms_archive = Path(sys.argv[5]).resolve()
    bundled_files = [Path(path).resolve() for path in sys.argv[6:]]
    dev_build = output.name.endswith("jdk_allmodules.zip")

    with tempfile.TemporaryDirectory(prefix = "bazel-package-") as temp:
        root = Path(temp)
        package_dir = root / "pkg"
        package_dir.mkdir()

        for source in bundled_files:
            shutil.copy2(source, package_dir / source.name)

        if dev_build:
            packaged_deploy_jar = deploy_jar
            label = "no_version"
        else:
            packaged_deploy_jar = root / "deploy-uncompressed.jar"
            label = _recompress_deploy_jar(deploy_jar, packaged_deploy_jar, root / "deploy")

        (package_dir / "build-label.txt").write_text(label, encoding = "utf-8")

        if embedded_tools_arg:
            embedded_tools = package_dir / "embedded_tools"
            embedded_tools.mkdir()
            _extract_zip(Path(embedded_tools_arg).resolve(), embedded_tools)

        _extract_tar(platforms_archive, package_dir)

        regular_files = sorted(
            path for path in package_dir.rglob("*") if path.is_file() and not path.is_symlink()
        )
        shutil.copy2(packaged_deploy_jar, package_dir / "A-server.jar")
        shutil.copy2(install_base_key, package_dir / "install_base_key")

        ordered_files = [package_dir / "A-server.jar"] + regular_files + [package_dir / "install_base_key"]
        compression_level = 1 if dev_build else 9
        with zipfile.ZipFile(
            output,
            "w",
            compression = zipfile.ZIP_DEFLATED,
            compresslevel = compression_level,
            allowZip64 = False,
        ) as archive:
            for path in ordered_files:
                archive.writestr(
                    _zip_info(path, path.relative_to(package_dir).as_posix(), zipfile.ZIP_DEFLATED),
                    path.read_bytes(),
                )


if __name__ == "__main__":
    main()
