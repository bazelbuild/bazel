# Packaging for Bazel

<div class="toc">
  <h2>Rules</h2>
  <ul>
    <li><a href="#pkg_tar">pkg_tar</a></li>
    <li><a href="#pkg_deb">pkg_deb</a></li>
  </ul>
</div>

## Overview

These build rules are used for building various packaging such as tarball
and debian package.

<a name="basic-example"></a>
## Basic Example

This example is a simplification of the debian packaging of Bazel:

```python

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar", "pkg_deb")

pkg_tar(
    name = "bazel-bin",
    strip_prefix = "/src",
    package_dir = "/usr/bin",
    files = ["//src:bazel"],
    mode = "0755",
)

pkg_tar(
    name = "bazel-tools",
    strip_prefix = "/",
    package_dir = "/usr/share/lib/bazel/tools",
    files = ["//tools:package-srcs"],
    mode = "0644",
    modes = {"tools/build_defs/docker/build_test.sh": "0755"},
)

pkg_tar(
    name = "debian-data",
    extension = "tar.gz",
    deps = [
        ":bazel-bin",
        ":bazel-tools",
    ],
)

pkg_deb(
    name = "bazel-debian",
    architecture = "amd64",
    built_using = "bazel (0.1.1)",
    data = ":debian-data",
    depends = [
        "zlib1g-dev",
        "unzip",
    ],
    description_file = "debian/description",
    homepage = "http://bazel.io",
    maintainer = "The Bazel Authors <bazel-dev@googlegroups.com>",
    package = "bazel",
    version = "0.1.1",
)
```

Here, the Debian package is built from three `pkg_tar` targets:

 - `bazel-bin` creates a tarball with the main binary (mode `0755`) in
   `/usr/bin`,
 - `bazel-tools` create a tarball with the base workspace (mode `0644`) to
   `/usr/share/bazel/tools` ; the `modes` attribute let us specifies executable
   files,
 - `debian-data` creates a gzip-compressed tarball that merge the three previous
   tarballs.

`debian-data` is then used for the data content of the debian archive created by
`pkg_deb`.

<a name="future"></a>
## Future work

 - Support more format, especially `pkg_zip`.
 - Maybe a bit more integration with the `docker_build` rule.

<a name="pkg_tar"></a>
## pkg_tar

```python
pkg_tar(name, extension, strip_prefix, package_dir, files,
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
            file will be '<i>name<i>.<i>extension<i>'. This extension
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
      <td><code>files</code></td>
      <td>
        <code>List of files, optional</code>
        <p>File to add to the layer.</p>
        <p>
          A list of files that should be included in the docker image.
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
           "tools/py/2to3.sh": "0755
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
  </tbody>
  </tbody>
</table>

<a name="pkg_deb"></a>
### pkg_deb

```python
pkg_deb(name, data, package, architecture, maintainer, preinst, postinst, prerm, postrm, version, version_file, description, description_file, built_using, built_using_file, priority, section, homepage, depends, suggests, enhances, conflicts, predepends, recommends)
```

Create a debian package. See <a
href="http://www.debian.org/doc/debian-policy/ch-controlfields.html">http://www.debian.org/doc/debian-policy/ch-controlfields.html</a>
for more details on this.

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
      <td><code>data</code></td>
      <td>
        <code>File, required</code>
        <p>
          A tar file that contains the data for the debian package (basically
          the list of files that will be installed by this package).
        </p>
      </td>
    </tr>
    <tr>
      <td><code>package</code></td>
      <td>
        <code>String, required</code>
        <p>The name of the package.</p>
      </td>
    </tr>
    <tr>
      <td><code>architecture</code></td>
      <td>
        <code>String, default to 'all'</code>
        <p>The architecture that this package target.</p>
        <p>
          See <a href="http://www.debian.org/ports/">http://www.debian.org/ports/</a>.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>maintainer</code></td>
      <td>
        <code>String, required</code>
        <p>The maintainer of the package.</p>
      </td>
    </tr>
    <tr>
      <td><code>preinst</code>, <code>postinst</code>, <code>prerm</code> and <code>postrm</code></td>
      <td>
        <code>Files, optional</code>
        <p>
          Respectively, the pre-install, post-install, pre-remove and
          post-remove scripts for the package.
        </p>
        <p>
          See <a href="http://www.debian.org/doc/debian-policy/ch-maintainerscripts.html">http://www.debian.org/doc/debian-policy/ch-maintainerscripts.html</a>.
      </td>
    </tr>
    <tr>
      <td><code>version</code>, <code>version_file</code></td>
      <td>
        <code>String or File, required</code>
        <p>
          The package version provided either inline (with <code>version</code>)
          or from a file (with <code>version_file</code>).
        </p>
      </td>
    </tr>
    <tr>
      <td><code>description</code>, <code>description_file</code></td>
      <td>
        <code>String or File, required</code>
        <p>
          The package description provided either inline (with <code>description</code>)
          or from a file (with <code>description_file</code>).
        </p>
      </td>
    </tr>
    <tr>
      <td><code>built_using</code>, <code>built_using_file</code></td>
      <td>
        <code>String or File, default to 'Bazel'</code>
        <p>
          The tool that were used to build this package provided either inline
          (with <code>built_using</code>) or from a file (with <code>built_using_file</code>).
        </p>
      </td>
    </tr>
    <tr>
      <td><code>priority</code></td>
      <td>
        <code>String, default to 'optional'</code>
        <p>The priority of the package.</p>
        <p>
          See <a href="http://www.debian.org/doc/debian-policy/ch-archive.html#s-priorities">http://www.debian.org/doc/debian-policy/ch-archive.html#s-priorities</a>.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>section</code></td>
      <td>
        <code>String, default to 'contrib/devel'</code>
        <p>The section of the package.</p>
        <p>
          See <a href="http://www.debian.org/doc/debian-policy/ch-archive.html#s-subsections">http://www.debian.org/doc/debian-policy/ch-archive.html#s-subsections</a>.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>homepage</code></td>
      <td>
        <code>String, optional</code>
        <p>The homepage of the project.</p>
      </td>
    </tr>
    <tr>
      <td>
        <code>depends</code>, <code>suggests</code>, <code>enhances</code>,
        <code>conflicts</code>, <code>predepends</code> and <code>recommends</code>.
      </td>
      <td>
        <code>String list, optional</code>
        <p>The list of dependencies in the project.</p>
        <p>
          See <a href="http://www.debian.org/doc/debian-policy/ch-relationships.html#s-binarydeps">http://www.debian.org/doc/debian-policy/ch-relationships.html#s-binarydeps</a>.
        </p>
      </td>
    </tr>
  </tbody>
  </tbody>
</table>
