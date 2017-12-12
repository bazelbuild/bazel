# Copyright 2016 The Bazel Authors. All rights reserved.
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

"""Quick and not really nice docker_pull rules based on the docker daemon."""

def can_user_docker(repository_ctx, docker):
  """Test whether current user is allowed to execute docker."""
  result = repository_ctx.execute([
    docker,
    "version",
  ])
  return result.return_code == 0


def _create_repo(repository_ctx):
  docker = repository_ctx.which("docker")

  user_enabled = False

  if docker:
    user_enabled = can_user_docker(repository_ctx, docker)
    if not repository_ctx.attr.optional:
      fail("Current user cannot execute docker commands")
    if not user_enabled:
      docker = None

  if docker == None and repository_ctx.attr.optional:
    if user_enabled:
      repository_ctx.file("BUILD", """
load("@io_bazel//tools/build_defs/docker:docker.bzl", "docker_build")

# an empty image to still allow building despite not having the base
# image.
docker_build(
    name = "image",
    visibility = ['//visibility:public'],
)
""")
    else:
      repository_ctx.file("BUILD")
      repository_ctx.file("enabled.bzl", """
CAN_TEST_WITH_DOCKER = False
""")
    return

  repository_ctx.file("BUILD", """
load("@io_bazel//tools/build_defs/docker:docker.bzl", "docker_build")

docker_build(
    name = "image",
    base = ":base.tar",
    visibility = ["//visibility:public"],
)
""")
  repository_ctx.file("enabled.bzl", """
CAN_TEST_WITH_DOCKER = True
""")
  return docker


def _config_impl(repository_ctx):
  _create_repo(repository_ctx)


def _pull_impl(repository_ctx):
  docker = _create_repo(repository_ctx)
  tag = repository_ctx.attr.tag
  cmd = "pull"
  if repository_ctx.attr.dockerfile:
    dockerfile = repository_ctx.path(repository_ctx.attr.dockerfile)
    cmd = "build"
    print("Running `docker build`")
    result = repository_ctx.execute([
        docker,
        "build",
        "-q",
        "-t",
        tag,
        "-f",
        dockerfile,
        dockerfile.dirname,
    ], quiet=False, timeout=3600)
  else:
    print("Running `docker pull`")
    result = repository_ctx.execute([docker, "pull", tag], quiet=False, timeout=3600)
  if result.return_code:
    fail("docker %s failed with error code %s:\n%s" % (
        cmd,
        result.return_code,
        result.stderr))
  result = repository_ctx.execute([
      docker, "save", "-o", repository_ctx.path("base.tar"), tag])
  if result.return_code:
    fail("docker save failed with error code %s:\n%s" % (
        result.return_code,
        result.stderr))

docker_config = repository_rule(
    implementation = _config_impl,
    attrs = {
        "optional": attr.bool(default=False),
    },
)


docker_pull = repository_rule(
    implementation = _pull_impl,
    attrs = {
        "tag": attr.string(mandatory=True),
        "dockerfile": attr.label(default=None),
        "optional": attr.bool(default=False),
    },
)
