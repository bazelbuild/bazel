# Copyright 2018 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Defines a repository rule that generates an archive consisting of the specified files to fetch"""

load("//:distdir_deps.bzl", "DEPS_BY_NAME")
load("//tools/build_defs/repo:http.bzl", "http_archive", "http_file", "http_jar")

_BUILD = """
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")

pkg_tar(
  name="archives",
  srcs = {srcs},
  package_dir = "{dirname}",
  visibility = ["//visibility:public"],
)

"""

def _distdir_tar_impl(ctx):
    for name in ctx.attr.archives:
        ctx.download(ctx.attr.urls[name], name, ctx.attr.sha256[name], False)
    ctx.file("WORKSPACE", "")
    ctx.file(
        "BUILD",
        _BUILD.format(srcs = ctx.attr.archives, dirname = ctx.attr.dirname),
    )

_distdir_tar_attrs = {
    "archives": attr.string_list(),
    "sha256": attr.string_dict(),
    "urls": attr.string_list_dict(),
    "dirname": attr.string(default = "distdir"),
}

_distdir_tar = repository_rule(
    implementation = _distdir_tar_impl,
    attrs = _distdir_tar_attrs,
)

def distdir_tar(name, archives, sha256, urls, dirname, dist_deps = None):
    """Creates a repository whose content is a set of tar files.

    Args:
      name: repo name.
      archives: list of tar file names.
      sha256: map of tar file names to SHAs.
      urls: map of tar file names to URL lists.
      dirname: output directory in repo.
      dist_deps: map of repo names to dict of archive, sha256, and urls.
    """
    if dist_deps:
        for dep, info in dist_deps.items():
            archive_file = info["archive"]
            archives.append(archive_file)
            sha256[archive_file] = info["sha256"]
            urls[archive_file] = info["urls"]
    _distdir_tar(
        name = name,
        archives = archives,
        sha256 = sha256,
        urls = urls,
        dirname = dirname,
    )

def dist_http_archive(name, **kwargs):
    """Wraps http_archive, providing attributes like sha and urls from the central list.

    dist_http_archive wraps an http_archive invocation, but looks up relevant attributes
    from distdir_deps.bzl so the user does not have to specify them.

    Args:
      name: repo name
      **kwargs: see http_archive for allowed args.
    """
    info = DEPS_BY_NAME[name]
    if "patch_args" not in kwargs:
        kwargs["patch_args"] = info.get("patch_args")
    if "patches" not in kwargs:
        kwargs["patches"] = info.get("patches")
    if "strip_prefix" not in kwargs:
        kwargs["strip_prefix"] = info.get("strip_prefix")
    http_archive(
        name = name,
        sha256 = info["sha256"],
        urls = info["urls"],
        **kwargs
    )

def dist_http_file(name, **kwargs):
    """Wraps http_file, providing attributes like sha and urls from the central list.

    dist_http_file wraps an http_file invocation, but looks up relevant attributes
    from distdir_deps.bzl so the user does not have to specify them.

    Args:
      name: repo name
      **kwargs: see http_file for allowed args.
    """
    info = DEPS_BY_NAME[name]
    http_file(
        name = name,
        sha256 = info["sha256"],
        urls = info["urls"],
        **kwargs
    )

def dist_http_jar(name, **kwargs):
    """Wraps http_jar, providing attributes like sha and urls from the central list.

    dist_http_jar wraps an http_jar invocation, but looks up relevant attributes
    from distdir_deps.bzl so the user does not have to specify them.

    Args:
      name: repo name
      **kwargs: see http_jar for allowed args.
    """
    info = DEPS_BY_NAME[name]
    http_jar(
        name = name,
        sha256 = info["sha256"],
        urls = info["urls"],
        **kwargs
    )
