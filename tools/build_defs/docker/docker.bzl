# Copyright 2015 The Bazel Authors. All rights reserved.
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
"""Rules for manipulation Docker images."""

# Filetype to restrict inputs
tar_filetype = FileType([".tar", ".tar.gz", ".tgz", ".tar.xz"])
deb_filetype = FileType([".deb"])

# Docker files are tarballs, should we allow other extensions than tar?
docker_filetype = tar_filetype

# This validates the two forms of value accepted by
# ENTRYPOINT and CMD, turning them into a canonical
# python list form.
#
# The Dockerfile construct:
#   ENTRYPOINT "/foo"
# Results in:
#   "Entrypoint": [
#       "/bin/sh",
#       "-c",
#       "\"/foo\""
#   ],
# Whereas:
#   ENTRYPOINT ["/foo", "a"]
# Results in:
#   "Entrypoint": [
#       "/foo",
#       "a"
#   ],
# NOTE: prefacing a command with 'exec' just ends up with the former
def _validate_command(name, argument):
  if type(argument) == "string":
    return ["/bin/sh", "-c", argument]
  elif type(argument) == "list":
    return argument
  elif argument:
    fail("The %s attribute must be a string or list, if specified." % name)
  else:
    return None

def _short_path_dirname(path):
  """Returns the directory's name of the short path of an artifact."""
  sp = path.short_path
  return sp[:sp.rfind("/")]

def _dest_path(f, strip_prefix):
  """Returns the short path of f, stripped of strip_prefix."""
  if not strip_prefix:
    # If no strip_prefix was specified, use the package of the
    # given input as the strip_prefix.
    strip_prefix = _short_path_dirname(f)
  if f.short_path.startswith(strip_prefix):
    return f.short_path[len(strip_prefix):]
  return f.short_path

def _compute_data_path(out, data_path):
  """Compute the relative data path prefix from the data_path attribute."""
  if data_path:
    # Strip ./ from the beginning if specified.
    # There is no way to handle .// correctly (no function that would make
    # that possible and Skylark is not turing complete) so just consider it
    # as an absolute path.
    if data_path[0:2] == "./":
      data_path = data_path[2:]
    if data_path[0] == "/":  # Absolute path
      return data_path[1:]
    elif not data_path or data_path == ".":  # Relative to current package
      return _short_path_dirname(out)
    else:  # Relative to a sub-directory
      return _short_path_dirname(out) + "/" + data_path
  return data_path

def _build_layer(ctx):
  """Build the current layer for appending it the base layer."""
  # Compute the relative path
  data_path = _compute_data_path(ctx.outputs.out, ctx.attr.data_path)

  layer = ctx.new_file(ctx.label.name + ".layer")
  build_layer = ctx.executable._build_layer
  args = [
      "--output=" + layer.path,
      "--directory=" + ctx.attr.directory,
      "--mode=" + ctx.attr.mode,
      ]
  args += ["--file=%s=%s" % (f.path, _dest_path(f, data_path))
           for f in ctx.files.files]
  args += ["--tar=" + f.path for f in ctx.files.tars]
  args += ["--deb=" + f.path for f in ctx.files.debs]
  args += ["--link=%s:%s" % (k, ctx.attr.symlinks[k])
           for k in ctx.attr.symlinks]

  ctx.action(
      executable = build_layer,
      arguments = args,
      inputs = ctx.files.files + ctx.files.tars + ctx.files.debs,
      outputs = [layer],
      mnemonic="DockerLayer"
      )
  return layer

def _sha256(ctx, artifact):
  """Create an action to compute the SHA-256 of an artifact."""
  out = ctx.new_file(artifact.basename + ".sha256")
  ctx.action(
      executable = ctx.executable._sha256,
      arguments = [artifact.path, out.path],
      inputs = [artifact],
      outputs = [out],
      mnemonic = "SHA256")
  return out

def _get_base_artifact(ctx):
  if ctx.files.base:
    if hasattr(ctx.attr.base, "docker_image"):
      return ctx.attr.base.docker_image
    if len(ctx.files.base) != 1:
      fail("base attribute should be a single tar file.")
    return ctx.files.base[0]

def _metadata_action(ctx, layer, name, output):
  """Generate the action to create the JSON metadata for the layer."""
  rewrite_tool = ctx.executable._rewrite_tool
  env = ctx.attr.env
  args = [
      "--output=%s" % output.path,
      "--layer=%s" % layer.path,
      "--name=@%s" % name.path,
      "--entrypoint=%s" % ",".join(ctx.attr.entrypoint),
      "--command=%s" % ",".join(ctx.attr.cmd),
      "--env=%s" % ",".join(["%s=%s" % (k, env[k]) for k in env]),
      "--ports=%s" % ",".join(ctx.attr.ports),
      "--volumes=%s" % ",".join(ctx.attr.volumes)
      ]
  inputs = [layer, rewrite_tool, name]
  base = _get_base_artifact(ctx)
  if base:
    args += ["--base=%s" % base.path]
    inputs += [base]

  ctx.action(
      executable = rewrite_tool,
      arguments = args,
      inputs = inputs,
      outputs = [output],
      mnemonic = "RewriteJSON")

def _compute_layer_name(ctx, layer):
  """Compute the layer's name.

  This function synthesize a version of its metadata where in place
  of its final name, we use the SHA256 of the layer blob.

  This makes the name of the layer a function of:
    - Its layer's SHA256
    - Its metadata
    - Its parent's name.
  Assuming the parent's name is derived by this same rigor, then
  a simple induction proves the content addressability.

  Args:
    ctx: Rule context.
    layer: The layer's artifact for which to compute the name.
  Returns:
    The artifact that will contains the name for the layer.
  """
  metadata = ctx.new_file(ctx.label.name + ".metadata-name")
  layer_sha = _sha256(ctx, layer)
  _metadata_action(ctx, layer, layer_sha, metadata)
  return _sha256(ctx, metadata)

def _metadata(ctx, layer, name):
  """Create the metadata for the new docker image."""
  metadata = ctx.new_file(ctx.label.name + ".metadata")
  _metadata_action(ctx, layer, name, metadata)
  return metadata

def _create_image(ctx, layer, name, metadata):
  """Create the new image."""
  create_image = ctx.executable._create_image
  args = [
      "--output=" + ctx.outputs.out.path,
      "--metadata=" + metadata.path,
      "--layer=" + layer.path,
      "--id=@" + name.path,
      # We label at push time, so we only put a single name in this file:
      #   bazel/package:target => {the layer being appended}
      # TODO(dmarting): Does the name makes sense? We could use the
      #   repositoryName/package instead. (why do we need to replace
      #   slashes?)
      "--repository=bazel/" + ctx.label.package.replace("/", "_"),
      "--name=" + ctx.label.name
      ]
  inputs = [layer, metadata, name]
  # If we have been provided a base image, add it.
  base = _get_base_artifact(ctx)
  if base:
    args += ["--base=%s" % base.path]
    inputs += [base]
  ctx.action(
      executable = create_image,
      arguments = args,
      inputs = inputs,
      use_default_shell_env = True,
      outputs = [ctx.outputs.out]
      )

def _docker_build_impl(ctx):
  """Implementation for the docker_build rule."""
  layer = _build_layer(ctx)
  name = _compute_layer_name(ctx, layer)
  metadata = _metadata(ctx, layer, name)
  _create_image(ctx, layer, name, metadata)
  ctx.file_action(
      content = "\n".join([
          "#!/bin/bash -eu",
          "docker load -i " + ctx.outputs.out.short_path
          ]),
      output = ctx.outputs.executable,
      executable = True)
  return struct(runfiles = ctx.runfiles(files = [ctx.outputs.out]),
                docker_image = ctx.outputs.out)

docker_build_ = rule(
    implementation = _docker_build_impl,
    attrs = {
        "base": attr.label(allow_files=docker_filetype),
        "data_path": attr.string(),
        "directory": attr.string(default="/"),
        "tars": attr.label_list(allow_files=tar_filetype),
        "debs": attr.label_list(allow_files=deb_filetype),
        "files": attr.label_list(allow_files=True),
        "mode": attr.string(default="0555"),
        "symlinks": attr.string_dict(),
        "entrypoint": attr.string_list(),
        "cmd": attr.string_list(),
        "env": attr.string_dict(),
        "ports": attr.string_list(),  # Skylark doesn't support int_list...
        "volumes": attr.string_list(),
        # Implicit dependencies.
        "_build_layer": attr.label(
            default=Label("//tools/build_defs/docker:build_layer"),
            cfg=HOST_CFG,
            executable=True,
            allow_files=True),
        "_create_image": attr.label(
            default=Label("//tools/build_defs/docker:create_image"),
            cfg=HOST_CFG,
            executable=True,
            allow_files=True),
        "_rewrite_tool": attr.label(
            default=Label("//tools/build_defs/docker:rewrite_json"),
            cfg=HOST_CFG,
            executable=True,
            allow_files=True),
        "_sha256": attr.label(
            default=Label("//tools/build_defs/docker:sha256"),
            cfg=HOST_CFG,
            executable=True,
            allow_files=True)
    },
    outputs = {
        "out": "%{name}.tar",
    },
    executable = True)

# Produces a new docker image tarball compatible with 'docker load', which
# is a single additional layer atop 'base'.  The goal is to have relatively
# complete support for building docker image, from the Dockerfile spec.
#
# Only 'name' is required. All other fields have sane defaults.
#
#   docker_build(
#      name="...",
#      visibility="...",
#
#      # The base layers on top of which to overlay this layer,
#      # equivalent to FROM.
#      base="//another/build:rule",]
#
#      # The base directory of the files, defaulted to
#      # the package of the input.
#      # All files structure relatively to that path will be preserved.
#      # A leading '/' mean the workspace root and this path is relative
#      # to the current package by default.
#      data_path="...",
#
#      # The directory in which to expand the specified files,
#      # defaulting to '/'.
#      # Only makes sense accompanying one of files/tars/debs.
#      directory="...",
#
#      # The set of archives to expand, or packages to install
#      # within the chroot of this layer
#      files=[...],
#      tars=[...],
#      debs=[...],
#
#      # The set of symlinks to create within a given layer.
#      symlinks = {
#          "/path/to/link": "/path/to/target",
#          ...
#      },
#
#      # https://docs.docker.com/reference/builder/#entrypoint
#      entrypoint="...", or
#      entrypoint=[...],            -- exec form
#
#      # https://docs.docker.com/reference/builder/#cmd
#      cmd="...", or
#      cmd=[...],                   -- exec form
#
#      # https://docs.docker.com/reference/builder/#expose
#      ports=[...],
#
#      # TODO(mattmoor): NYI
#      # https://docs.docker.com/reference/builder/#maintainer
#      maintainer="...",
#
#      # TODO(mattmoor): NYI
#      # https://docs.docker.com/reference/builder/#user
#      # NOTE: the normal directive affects subsequent RUN, CMD,
#      # and ENTRYPOINT
#      user="...",
#
#      # https://docs.docker.com/reference/builder/#volume
#      volumes=[...],
#
#      # TODO(mattmoor): NYI
#      # https://docs.docker.com/reference/builder/#workdir
#      # NOTE: the normal directive affects subsequent RUN, CMD,
#      # ENTRYPOINT, ADD, and COPY
#      workdir="...",
#
#      # https://docs.docker.com/reference/builder/#env
#      env = {
#         "var1": "val1",
#         "var2": "val2",
#         ...
#         "varN": "valN",
#      },
#
#      # NOTE: Without a motivating use case, there is little reason to support:
#      # https://docs.docker.com/reference/builder/#onbuild
#   )
def docker_build(**kwargs):
  """Package a docker image.

  This rule generates a sequence of genrules the last of which is named 'name',
  so the dependency graph works out properly.  The output of this rule is a
  tarball compatible with 'docker save/load' with the structure:
    {layer-name}:
      layer.tar
      VERSION
      json
    ...
    repositories
    top     # an implementation detail of our rules, not consumed by Docker.
  This rule appends a single new layer to the tarball of this form provided
  via the 'base' parameter.

  The images produced by this rule are always named 'blaze/tmp:latest' when
  loaded (an internal detail).  The expectation is that the images produced
  by these rules will be uploaded using the 'docker_push' rule below.

  Args:
    **kwargs: See above.
  """
  if "cmd" in kwargs:
    kwargs["cmd"] = _validate_command("cmd", kwargs["cmd"])
  if "entrypoint" in kwargs:
    kwargs["entrypoint"] = _validate_command("entrypoint", kwargs["entrypoint"])
  docker_build_(**kwargs)
