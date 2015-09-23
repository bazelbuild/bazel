# Docker support for Bazel

## Overview

These build rules are used for building [Docker](https://www.docker.com)
images. Such images are easy to modify and deploy system image for
deploying application easily on cloud providers.

As traditional Dockerfile-based `docker build`s effectively execute a series
of commands inside of Docker containers, saving the intermediate results as
layers; this approach is unsuitable for use in Bazel for a variety of reasons.

The docker_build rule constructs a tarball that is compatible with
`docker save/load`, and creates a single layer out of each BUILD rule in the chain.

* [Basic Example](#basic-example)
* [Build Rule Reference](#reference)
  * [`docker_build`](#docker_build)
* [Future work](#future)

<a name="basic-example"></a>
## Basic Example

Consider the following BUILD file in `//third_party/debian`:

```python
filegroup(
    name = "ca_certificates",
    srcs = ["ca_certificates.deb"],
)

# Example when you have all your dependencies in your repository.
# We have an example on how to fetch them from the web later in this
# document.
filegroup(
    name = "openjdk-7-jre-headless",
    srcs = ["openjdk-7-jre-headless.deb"],
)

docker_build(
    name = "wheezy",
    tars = ["wheezy.tar"],
)
```

The `wheezy` target in that BUILD file roughly corresponds to the Dockerfile:

```docker
FROM scratch
ADD wheezy.tar /
```

You can then build up subsequent layers via:

```python
docker_build(
    name = "base",
    base = "//third_party/debian:wheezy",
    debs = ["//third_party/debian:ca_certificates"],
)

docker_build(
    name = "java",
    base = ":base",
    debs = ["//third_party/debian:openjdk-7-jre-headless"],
)
```

## Metadata

You can set layer metadata on these same rules by simply adding (supported) arguments to the rule, for instance:

```python
docker_build(
    name = "my-layer",
    entrypoint = ["foo", "bar", "baz"],
    ...
)
```

Will have a similar effect as the Dockerfile construct:

```docker
ENTRYPOINT ["foo", "bar", "baz"]
```

For the set of supported metadata, and ways to construct layers, see here.


### Using

Suppose you have a `docker_build` target `//my/image:helloworld`:

```python
docker_build(
    name = "helloworld",
    ...
)
```

You can build this with `bazel build my/image:helloworld`.
This will produce the file `bazel-genfiles/my/image/helloworld.tar`.
You can load this into my local Docker client by running
`docker load -i bazel-genfiles/my/image/helloworld.tar`, or simply
`bazel run my/image:helloworld`.


Upon success you should be able to run `docker images` and see:

```
REPOSITORY          TAG                 IMAGE ID       ...
bazel/my_image      helloworld          d3440d7f2bde   ...
```

You can now use this docker image with the name `bazel/my_image:helloworld` or
tag it with another name, for example:
`docker tag bazel/my_image:helloworld gcr.io/my-project/my-awesome-image:v0.9`

__Nota Bene:__ the `docker images` command will show a really old timestamp
because `docker_build` remove all timestamps from the build to make it
reproducible.

## Pulling images and deb files from the internet

If you do not want to check in base image in your repository, you can use
[external repositories](http://bazel.io/docs/external.html). For instance,
you could create various layer with `external` labels:

```python
load("/tools/build_defs/docker/docker", "docker_build")

docker_build(
    name = "java",
    base = "@docker-debian//:wheezy",
    debs = ["@openjdk-7-jre-headless//file"],
)
```

Using the WORKSPACE file to add the actual files:

```python
new_http_archive(
    name = "docker-debian",
    url = "https://codeload.github.com/tianon/docker-brew-debian/zip/e9bafb113f432c48c7e86c616424cb4b2f2c7a51",
    build_file = "debian.BUILD",
    type = "zip",
    sha256 = "515d385777643ef184729375bc5cb996134b3c1dc15c53acf104749b37334f68",
)

http_file(
   name = "openjdk-7-jre-headless",
   url = "http://security.debian.org/debian-security/pool/updates/main/o/openjdk-7/openjdk-7-jre-headless_7u79-2.5.5-1~deb7u1_amd64.deb",
   sha256 = "b632f0864450161d475c012dcfcc37a1243d9ebf7ff9d6292150955616d71c23",
)
```

With the following `debian.BUILD` file:

```python
load("/tools/build_defs/docker/docker", "docker_build")

# Extract .xz files
genrule(
    name = "wheezy_tar",
    srcs = ["docker-brew-debian-e9bafb113f432c48c7e86c616424cb4b2f2c7a51/wheezy/rootfs.tar.xz"],
    outs = ["wheezy_tar.tar"],
    cmd = "cat $< | xzcat >$@",
)

docker_build(
    name = "wheezy",
    tars = [":wheezy_tar"],
    visibility = ["//visibility:public"],
)
```

<a name="reference"></a>
## Build Rule Reference [reference]

<a name="docker_build"></a>
### `docker_build`

`docker_build(name, base, data_path, directory, files, tars, debs,
symlinks, entrypoint, cmd, env, ports, volumes)`

<table>
  <thead>
    <tr>
      <th>Attribute</th>
      <th>Description</th>
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
      <td><code>base</code></td>
      <td>
        <code>File, optional</code>
        <p>
            The base layers on top of which to overlay this layer, equivalent to
            FROM.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>data_path</code></td>
      <td>
        <code>String, optional</code>
        <p>Root path of the files.</p>
        <p>
          The directory structure from the files is preserved inside the
          docker image but a prefix path determined by `data_path`
          is removed from the directory structure. This path can
          be absolute from the workspace root if starting with a `/` or
          relative to the rule's directory. A relative path may starts with "./"
          (or be ".") but cannot use go up with "..". By default, the
          `data_path` attribute is unused and all files are supposed to have no
          prefix.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>directory</code></td>
      <td>
        <code>String, optional</code>
        <p>Target directory.</p>
        <p>
          The directory in which to expand the specified files, defaulting to '/'.
          Only makes sense accompanying one of files/tars/debs.
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
      <td><code>tars</code></td>
      <td>
        <code>List of files, optional</code>
        <p>Tar file to extract in the layer.</p>
        <p>
          A list of tar files whose content should be in the docker image.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>debs</code></td>
      <td>
        <code>List of files, optional</code>
        <p>Debian package to install.</p>
        <p>
          A list of debian packages that will be installed in the docker image.
        </p>
      </td>
    </tr>
    <tr>
      <td><code>symlinks</code></td>
      <td>
        <code>Dictionary, optional</code>
        <p>Symlinks to create in the docker image.</p>
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
      <td><code>entrypoint</code></td>
      <td>
        <code>String or string list, optional</code>
        <p><a href="https://docs.docker.com/reference/builder/#entrypoint">List
               of entrypoints to add in the layer.</a></p>
      </td>
    </tr>
    <tr>
      <td><code>cmd</code></td>
      <td>
        <code>String or string list, optional</code>
        <p><a href="https://docs.docker.com/reference/builder/#cmd">List
               of commands to execute in the layer.</a></p>
      </td>
    </tr>
    <tr>
      <td><code>ports</code></td>
      <td>
        <code>String list, optional</code>
        <p><a href="https://docs.docker.com/reference/builder/#expose">List
               of ports to expose.</a></p>
      </td>
    </tr>
    <tr>
      <td><code>volumes</code></td>
      <td>
        <code>String list, optional</code>
        <p><a href="https://docs.docker.com/reference/builder/#volumes">List
               of volumes to mount.</a></p>
      </td>
    </tr>
  </tbody>
  </tbody>
</table>

<a name="future"></a>
# Future work

In the future, we would like to provide better integration with docker
repositories: pull and push docker image.
