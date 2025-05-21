Project: /_project.yaml
Book: /_book.yaml

# Bazel registries

{% include "_buttons.html" %}

Bazel discovers dependencies by requesting their information from Bazel
*registries*: databases of Bazel modules. Bazel only supports one type of
registries — [*index registries*](#index_registry) — local directories or static
HTTP servers following a specific format.

## Index registry

An index registry is a local directory or a static HTTP server containing
information about a list of modules — including their homepage, maintainers, the
`MODULE.bazel` file of each version, and how to fetch the source of each
version. Notably, it does *not* need to serve the source archives itself.

An index registry must have the following format:

*   [`/bazel_registry.json`](#bazel-registry-json): An optional JSON file
    containing metadata for the registry.
*   `/modules`: A directory containing a subdirectory for each module in this
    registry
*   `/modules/$MODULE`: A directory containing a subdirectory for each version
    of the module named `$MODULE`, as well as the [`metadata.json`
    file](#metadata-json) containing metadata for this module.
*   `/modules/$MODULE/$VERSION`: A directory containing the following files:
    *   `MODULE.bazel`: The `MODULE.bazel` file of this module version. Note
        that this is the `MODULE.bazel` file read during Bazel's external
        dependency resolution, _not_ the one from the source archive (unless
        there's a non-registry override).
    *   [`source.json`](#source-json): A JSON file containing information on how
        to fetch the source of this module version
    *   `patches/`: An optional directory containing patch files, only used when
        `source.json` has "archive" type
    *   `overlay/`: An optional directory containing overlay files, only used
        when `source.json` has "archive" type

### `bazel_registry.json` {:#bazel-registry-json}

`bazel_registry.json` is an optional file that specifies metadata applying to
the entire registry. It can contain the following fields:

*   `mirrors`: an array of strings, specifying the list of mirrors to use for
    source archives.
    *   The mirrored URL is a concatenation of the mirror itself, and the
        source URL of the module specified by its `source.json` file sans the
        protocol. For example, if a module's source URL is
        `https://foo.com/bar/baz`, and `mirrors` contains
        `["https://mirror1.com/", "https://example.com/mirror2/"]`, then the
        URLs Bazel will try in order are `https://mirror1.com/foo.com/bar/baz`,
        `https://example.com/mirror2/foo.com/bar/baz`, and finally the original
        source URL itself `https://foo.com/bar/baz`.
*   `module_base_path`: a string, specifying the base path for modules with
    `local_path` type in the `source.json` file

### `metadata.json` {:#metadata-json}

`metadata.json` is an optional JSON file containing information about the
module, with the following fields:

*   `versions`: An array of strings, each denoting a version of the module
    available in this registry. This array should match the children of the
    module directory.
*   `yanked_versions`: A JSON object specifying the [*yanked*
    versions](/external/module#yanked_versions) of this module. The keys
    should be versions to yank, and the values should be descriptions of
    why the version is yanked, ideally containing a link to more
    information.

Note that the BCR requires more information in the `metadata.json` file.

### `source.json` {:#source-json}

`source.json` is a required JSON file containing information about how to fetch
a specific version of a module. The schema of this file depends on its `type`
field, which defaults to `archive`.

*   If `type` is `archive` (the default), this module version is backed by an
    [`http_archive`](/rules/lib/repo/http#http_archive) repo rule; it's fetched
    by downloading an archive from a given URL and extracting its contents. It
    supports the following fields:
    *   `url`: A string, the URL of the source archive
    *   `mirror_urls`: A list of string, the mirror URLs of the source archive.
        The URLs are tried in order after `url` as backups.
    *   `integrity`: A string, the [Subresource
        Integrity][subresource-integrity] checksum of the archive
    *   `strip_prefix`: A string, the directory prefix to strip when extracting
        the source archive
    *   `overlay`: A JSON object containing overlay files to layer on top of the
        extracted archive. The patch files are located under the
        `/modules/$MODULE/$VERSION/overlay` directory. The keys are the
        overlay file names, and the values are the integrity checksum of
        the overlay files. The overlays are applied before the patch files.
    *   `patches`: A JSON object containing patch files to apply to the
        extracted archive. The patch files are located under the
        `/modules/$MODULE/$VERSION/patches` directory. The keys are the
        patch file names, and the values are the integrity checksum of
        the patch files. The patches are applied after the overlay files and in
        the order they appear in `patches`.
    *   `patch_strip`: A number; the same as the `--strip` argument of Unix
        `patch`.
    *   `archive_type`: A string, the archive type of the downloaded file (Same
        as [`type` on `http_archive`](/rules/lib/repo/http#http_archive-type)).
*   If `type` is `git_repository`, this module version is backed by a
    [`git_repository`](/rules/lib/repo/git#git_repository) repo rule; it's
    fetched by cloning a Git repository.
    *   The following fields are supported, and are directly forwarded to the
        underlying `git_repository` repo rule: `remote`, `commit`,
        `shallow_since`, `tag`, `init_submodules`, `verbose`, and
        `strip_prefix`.
*   If `type` is `local_path`, this module version is backed by a
    [`local_repository`](/rules/lib/repo/local#local_repository) repo rule;
    it's symlinked to a directory on local disk. It supports the following
    field:
    *   `path`: The local path to the repo, calculated as following:
        *   If `path` is an absolute path, it stays as it is
        *   If `path` is a relative path and `module_base_path` is an
            absolute path, it resolves to `<module_base_path>/<path>`
        *   If `path` and `module_base_path` are both relative paths, it
            resolves to `<registry_path>/<module_base_path>/<path>`.
            Registry must be hosted locally and used by
            `--registry=file://<registry_path>`. Otherwise, Bazel will
            throw an error

## Bazel Central Registry {:#bazel-central-registry}

The Bazel Central Registry (BCR) at <https://bcr.bazel.build/> is an index
registry with contents backed by the GitHub repo
[`bazelbuild/bazel-central-registry`][bcr-repo]. You can browse its contents
using the web frontend at <https://registry.bazel.build/>.

The Bazel community maintains the BCR, and contributors are welcome to submit
pull requests. See the [BCR contribution
guidelines][bcr-contribution-guidelines].

In addition to following the format of a normal index registry, the BCR requires
a `presubmit.yml` file for each module version
(`/modules/$MODULE/$VERSION/presubmit.yml`). This file specifies a few essential
build and test targets that you can use to check the validity of this module
version. The BCR's CI pipelines also uses this to ensure interoperability
between modules.

## Selecting registries

The repeatable Bazel flag `--registry` can be used to specify the list of
registries to request modules from, so you can set up your project to fetch
dependencies from a third-party or internal registry. Earlier registries take
precedence. For convenience, you can put a list of `--registry` flags in the
`.bazelrc` file of your project.

If your registry is hosted on GitHub (for example, as a fork of
`bazelbuild/bazel-central-registry`) then your `--registry` value needs a raw
GitHub address under `raw.githubusercontent.com`. For example, on the `main`
branch of the `my-org` fork, you would set
`--registry=https://raw.githubusercontent.com/my-org/bazel-central-registry/main/`.

Using the `--registry` flag stops the Bazel Central Registry from being used by
default, but you can add it back by adding `--registry=https://bcr.bazel.build`.

[bcr-contribution-guidelines]: https://github.com/bazelbuild/bazel-central-registry/blob/main/docs/README.md
[bcr-repo]: https://github.com/bazelbuild/bazel-central-registry
[subresource-integrity]: https://w3c.github.io/webappsec-subresource-integrity/#integrity-metadata-description
