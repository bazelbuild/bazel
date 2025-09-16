Project: /_project.yaml
Book: /_book.yaml

# Bazel registries

{% include "_buttons.html" %}

Bzlmod discovers dependencies by requesting their information from Bazel
*registries*: databases of Bazel modules. Currently, Bzlmod only supports
[*index registries*](#index_registry) — local directories or static HTTP servers
following a specific format.

## Index registry

An index registry is a local directory or a static HTTP server containing
information about a list of modules — including their homepage, maintainers, the
`MODULE.bazel` file of each version, and how to fetch the source of each
version. Notably, it does *not* need to serve the source archives itself.

An index registry must follow the format below:

*   `/bazel_registry.json`: A JSON file containing metadata for the registry
    like:
    *   `mirrors`: specifying the list of mirrors to use for source archives.
        The mirrored URL is a concatenation of the mirror itself, and the
        source URL of the module specified by its `source.json` file sans the
        protocol. For example, if a module's source URL is
        `https://foo.com/bar/baz`, and `mirrors` contains
        `["https://mirror1.com/", "https://example.com/mirror2/"]`, then the
        URLs Bazel will try in order are `https://mirror1.com/foo.com/bar/baz`,
        `https://example.com/mirror2/foo.com/bar/baz`, and finally the original
        source URL itself `https://foo.com/bar/baz`.
    *   `module_base_path`: specifying the base path for modules with
        `local_repository` type in the `source.json` file
*   `/modules`: A directory containing a subdirectory for each module in this
    registry
*   `/modules/$MODULE`: A directory containing a subdirectory for each version
    of this module, as well as:
    *   `metadata.json`: A JSON file containing information about the module,
        with the following fields:
        *   `homepage`: The URL of the project's homepage
        *   `maintainers`: A list of JSON objects, each of which corresponds to
            the information of a maintainer of the module *in the registry*.
            Note that this is not necessarily the same as the *authors* of the
            project
        *   `versions`: A list of all the versions of this module to be found in
            this registry
        *   `yanked_versions`: A map of [*yanked*
            versions](/external/module#yanked_versions) of this module. The keys
            should be versions to yank and the values should be descriptions of
            why the version is yanked, ideally containing a link to more
            information
*   `/modules/$MODULE/$VERSION`: A directory containing the following files:
    *   `MODULE.bazel`: The `MODULE.bazel` file of this module version
    *   `source.json`: A JSON file containing information on how to fetch the
        source of this module version
        *   The default type is "archive", representing an `http_archive` repo,
            with the following fields:
            *   `url`: The URL of the source archive
            *   `integrity`: The [Subresource
                Integrity](https://w3c.github.io/webappsec-subresource-integrity/#integrity-metadata-description){: .external}
                checksum of the archive
            *   `strip_prefix`: A directory prefix to strip when extracting the
                source archive
            *   `patches`: A map containing patch files to apply to the
                extracted archive. The patch files are located under the
                `/modules/$MODULE/$VERSION/patches` directory. The keys are the
                patch file names, and the values are the integrity checksum of
                the patch files
            *   `patch_strip`: Same as the `--strip` argument of Unix `patch`.
            *   `archive_type`: The archive type of the downloaded file (Same as `type` on `http_archive`).
                By default, the archive type is determined from the file extension of the URL. If the file has
                no extension, you can explicitly specify one of the following: `"zip"`, `"jar"`, `"war"`, `"aar"`,
                `"tar"`, `"tar.gz"`, `"tgz"`, `"tar.xz"`, `"txz"`, `"tar.zst"`, `"tzst"`, `tar.bz2`, `"ar"`, or `"deb"`.
        *   The type can be changed to use a git repository, with these fields:
            *   `type`: `git_repository`
            *   The following fields as described at https://bazel.build/rules/lib/repo/git:
                * `remote`
                * `commit`
                * `shallow_since`
                * `tag`
                * `init_submodules`
                * `verbose`
                * `strip_prefix`
        *   The type can be changed to use a local path, representing a
            `local_repository` repo, with these fields:
            *   `type`: `local_path`
            *   `path`: The local path to the repo, calculated as following:
                *   If `path` is an absolute path, it stays as it is
                *   If `path` is a relative path and `module_base_path` is an
                    absolute path, it resolves to `<module_base_path>/<path>`
                *   If `path` and `module_base_path` are both relative paths, it
                    resolves to `<registry_path>/<module_base_path>/<path>`.
                    Registry must be hosted locally and used by
                    `--registry=file://<registry_path>`. Otherwise, Bazel will
                    throw an error
    *   `patches/`: An optional directory containing patch files, only used when
        `source.json` has "archive" type

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
        the patch files. The patches are applied after the overlay files.
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
[`bazelbuild/bazel-central-registry`](https://github.com/bazelbuild/bazel-central-registry){: .external}.
You can browse its contents using the web frontend at
<https://registry.bazel.build/>.

The Bazel community maintains the BCR, and contributors are welcome to submit
pull requests. See the [BCR contribution
guidelines](https://github.com/bazelbuild/bazel-central-registry/blob/main/docs/README.md){: .external}.

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
