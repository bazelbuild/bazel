# Packaging for Bazel

## Deprecated

These rules have been extracted from the Bazel sources and are now available at
[bazelbuild/rules_pkg](https://github.com/bazelbuild/rules_pkg/releases)
 [(docs)](https://github.com/bazelbuild/rules_pkg/tree/main/pkg).

Issues and PRs against the built-in versions of these rules will no longer be
addressed. This page will exist for reference until the code is removed from
Bazel.

For more information, follow [issue 8857](https://github.com/bazelbuild/bazel/issues/8857)

## rules_pkg

<div class="toc">
  <h2>Rules</h2>
  <ul>
    <li><a href="#pkg_tar">pkg_tar</a></li>
  </ul>
</div>

## Overview

`pkg_tar()` is available for building a .tar file without depending
on anything besides Bazel. Since this feature is deprecated and will
eventually be removed from Bazel, you should migrate to `@rules_pkg`.

<a name="basic-example"></a>
## Basic Example

This example is a simplification of building Bazel and creating a distribution
tarball.

```python
load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "bazel-bin",
    strip_prefix = "/src",
    package_dir = "/usr/bin",
    srcs = ["//src:bazel"],
    mode = "0755",
)

pkg_tar(
    name = "bazel-tools",
    strip_prefix = "/",
    package_dir = "/usr/share/lib/bazel/tools",
    srcs = ["//tools:package-srcs"],
    mode = "0644",
)

pkg_tar(
    name = "bazel-all",
    extension = "tar.gz",
    deps = [
        ":bazel-bin",
        ":bazel-tools",
    ],
)
```

Here, a package is built from three `pkg_tar` targets:

 - `bazel-bin` creates a tarball with the main binary (mode `0755`) in
   `/usr/bin`,
 - `bazel-tools` create a tarball with the base workspace (mode `0644`) to
   `/usr/share/bazel/tools` ; the `modes` attribute let us specifies executable
   files,
 - `bazel-all` creates a gzip-compressed tarball that merge the two previous
   tarballs.


<a name="pkg_tar"></a>
## pkg_tar

```python
pkg_tar(name, extension, strip_prefix, package_dir, srcs,
mode, modes, deps, symlinks)
```

Creates a tar file from a list of inputs.

<table class="table table-condensed table-bordered table-params">
  <colgroup>
    <col class="col-param" />
    <col class="param-description" />
  </colgroup>
  <thead>
    <tr>
      <th colspan="2">Attributes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>name</code></td>
      <td>
        <code>Name, required</code>
        <p>A unique name for this rule.</p>
      </td>
    </tr>
    <tr>
      <td><code>extension</code></td>
      <td>
        <code>String, default to 'tar'</code>
        <p>
            The extension for the resulting tarball. The output
            file will be '<i>name</i>.<i>extension</i>'. This extension
            also decide on the compression: if set to <code>tar.gz</code>
            or <code>tgz</code> then gzip compression will be used and
            if set to <code>tar.bz2</code> or <code>tar.bzip2</code> then
            bzip2 compression will be used.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>strip_prefix</code></td>
      <td>
        <code>String, optional</code>
        <p>Root path of the files.</p>
        <p>
          The directory structure from the files is preserved inside the
          tarball but a prefix path determined by <code>strip_prefix</code>
          is removed from the directory structure. This path can
          be absolute from the workspace root if starting with a <code>/</code> or
          relative to the rule's directory. A relative path may start with "./"
          (or be ".") but cannot use ".." to go up level(s). By default, the
          <code>strip_prefix</code> attribute is unused and all files are supposed to have no
          prefix. A <code>strip_prefix</code> of "" (the empty string) means the
          same as the default.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>package_dir</code></td>
      <td>
        <code>String, optional</code>
        <p>Target directory.</p>
        <p>
          The directory in which to expand the specified files, defaulting to '/'.
          Only makes sense accompanying files.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>srcs</code></td>
      <td>
        <code>List of files, optional</code>
        <p>File to add to the layer.</p>
        <p>
          A list of files that should be included in the archive.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>mode</code></td>
      <td>
        <code>String, default to 0555</code>
        <p>
          Set the mode of files added by the <code>files</code> attribute.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>mtime</code></td>
      <td>
        <code>int, seconds since Jan 1, 1970, default to -1 (ignored)</code>
        <p>
          Set the mod time of files added by the <code>files</code> attribute.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>portable_mtime</code></td>
      <td>
        <code>bool, default True</code>
        <p>
          Set the mod time of files added by the <code>files</code> attribute
          to a 2000-01-01.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>modes</code></td>
      <td>
        <code>Dictionary, default to '{}'</code>
        <p>
          A string dictionary to change default mode of specific files from
          <code>files</code>. Each key should be a path to a file before
          appending the prefix <code>package_dir</code> and the corresponding
          value the octal permission of to apply to the file.
        </p>
        <p>
          <code>
          modes = {
           "tools/py/2to3.sh": "0755",
           ...
          },
          </code>
        </p>
      </td>
    </tr>
    <tr>
      <td><code>owner</code></td>
      <td>
        <code>String, default to '0.0'</code>
        <p>
          <code>UID.GID</code> to set the default numeric owner for all files
          provided in <code>files</code>.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>owners</code></td>
      <td>
        <code>Dictionary, default to '{}'</code>
        <p>
          A string dictionary to change default owner of specific files from
          <code>files</code>. Each key should be a path to a file before
          appending the prefix <code>package_dir</code> and the corresponding
          value the <code>UID.GID</code> numeric string for the owner of the
          file. When determining owner ids, this attribute is looked first then
          <code>owner</code>.
        </p>
        <p>
          <code>
          owners = {
           "tools/py/2to3.sh": "42.24",
           ...
          },
          </code>
        </p>
      </td>
    </tr>
    <tr>
      <td><code>ownername</code></td>
      <td>
        <code>String, optional</code>
        <p>
          <code>username.groupname</code> to set the default owner for all files
          provided in <code>files</code> (by default there is no owner names).
        </p>
      </td>
    </tr>
    <tr>
      <td><code>ownernames</code></td>
      <td>
        <code>Dictionary, default to '{}'</code>
        <p>
          A string dictionary to change default owner of specific files from
          <code>files</code>. Each key should be a path to a file before
          appending the prefix <code>package_dir</code> and the corresponding
          value the <code>username.groupname</code> string for the owner of the
          file. When determining ownernames, this attribute is looked first then
          <code>ownername</code>.
        </p>
        <p>
          <code>
          owners = {
           "tools/py/2to3.sh": "leeroy.jenkins",
           ...
          },
          </code>
        </p>
      </td>
    </tr>
    <tr>
      <td><code>deps</code></td>
      <td>
        <code>List of labels, optional</code>
        <p>Tar files to extract and include in this tar package.</p>
        <p>
          A list of tarball labels to merge into the output tarball.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>symlinks</code></td>
      <td>
        <code>Dictionary, optional</code>
        <p>Symlinks to create in the output tarball.</p>
        <p>
          <code>
          symlinks = {
           "/path/to/link": "/path/to/target",
           ...
          },
          </code>
        </p>
      </td>
    </tr>
    <tr>
      <td><code>remap_paths</code></td>
      <td>
        <code>Dictionary, optional</code>
        <p>Source path prefixes to remap in the tarfile.</p>
        <p>
          <code>
          remap_paths = {
           "original/path/prefix": "replaced/path",
           ...
          },
          </code>
        </p>
      </td>
    </tr>
  </tbody>
</table>
