---
layout: posts
title: Building deterministic Docker images with Bazel
---

[Docker](https://www.docker.com) images are great to automate your deployment
environment. By composing base images, you can create an (almost) reproducible
environment and, using an appropriate cloud service, easily deploy those
image. However, V1 Docker build suffers several issues:

  1. Docker images are non-hermetic as they can run any command,
  2. Docker images are non-reproducible: each "layer" identifier is a **random**
  hex string (and not cryptographic hash of the layer content), and
  3. Docker image builds are not incremental since Docker assumes that `RUN foo`
  always does the same thing.

Googlers working on [Google Container Registry](https://gcr.io) developed a support
for building reproducible Docker images using Skylark / Bazel that address these
problems. We recently [shipped](https://github.com/google/bazel/commit/5f25891bb17d19cb1208ddad1e88cc4bb4a56782)
it.

Of course, it does not support `RUN` command, but the rule also strips
timestamps of the tar file and use a SHA sum that is function of the layer
data as layer identifier. This ensure reproducibility and correct
incrementality.

To use it, simply creates your images using the BUILD language:

```python
load("/tools/build_defs/docker/docker", "docker_build")

docker_build(
   name = “foo”,
   tars = [ “base.tar” ],
)

docker_build(
   name = “bar”,
   base = “:foo”,
   debs = [ “blah.deb” ],
   files = [ “:bazinga” ],
   volumes = [ “/asdf” ],
)
```

This will generate two docker images loadable with `bazel run :foo` and `bazel
run :bar`. The `foo` target is roughly equivalent to the following Dockerfile:

```
FROM bazel/base
```

And the `bar` target is roughly equivalent to the following Dockerfile:
```
FROM bazel/foo
RUN dpkg -i blah.deb
ADD bazinga /
VOLUMES /asdf
```

Using [remote repositories](http://bazel.io/docs/external.html), it is possible
to fetch the various base image for the web and we are working on providing a
`docker_pull` rule to interact more fluently with existing images.

You can learn more about this docker support
[here](https://github.com/google/bazel/blob/master/tools/build_defs/docker/README.md).
