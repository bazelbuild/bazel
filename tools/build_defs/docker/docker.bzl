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
tar_filetype = [".tar", ".tar.gz", ".tgz", ".tar.xz"]
deb_filetype = [".deb", ".udeb"]

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
    # as an absolute path. A data_path of / should preserve the entire
    # path up to the repository root.
    if data_path == "/":
      return data_path
    if len(data_path) >= 2 and data_path[0:2] == "./":
      data_path = data_path[2:]
    if not data_path or data_path == ".":  # Relative to current package
      return _short_path_dirname(out)
    elif data_path[0] == "/":  # Absolute path
      return data_path[1:]
    else:  # Relative to a sub-directory
      return _short_path_dirname(out) + "/" + data_path
  return data_path

def _build_layer(ctx):
  """Build the current layer for appending it the base layer."""
  # Compute the relative path
  data_path = _compute_data_path(ctx.outputs.out, ctx.attr.data_path)

  layer = ctx.new_file(ctx.label.name + ".layer")
  build_layer = ctx.executable.build_layer
  args = [
      "--output=" + layer.path,
      "--directory=" + ctx.attr.directory,
      "--mode=" + ctx.attr.mode,
      ]
  args += ["--file=%s=%s" % (f.path, _dest_path(f, data_path))
           for f in ctx.files.files]
  args += ["--tar=" + f.path for f in ctx.files.tars]
  args += ["--deb=" + f.path for f in ctx.files.debs if f.path.endswith(".deb")]
  args += ["--link=%s:%s" % (k, ctx.attr.symlinks[k])
           for k in ctx.attr.symlinks]
  arg_file = ctx.new_file(ctx.label.name + ".layer.args")
  ctx.file_action(arg_file, "\n".join(args))

  ctx.action(
      executable = build_layer,
      arguments = ["--flagfile=" + arg_file.path],
      inputs = ctx.files.files + ctx.files.tars + ctx.files.debs + [arg_file],
      outputs = [layer],
      use_default_shell_env=True,
      mnemonic="DockerLayer"
      )
  return layer

def _sha256(ctx, artifact):
  """Create an action to compute the SHA-256 of an artifact."""
  out = ctx.new_file(artifact.basename + ".sha256")
  ctx.action(
      executable = ctx.executable.sha256,
      arguments = [artifact.path, out.path],
      inputs = [artifact],
      outputs = [out],
      mnemonic = "SHA256")
  return out

def _get_base_artifact(ctx):
  if ctx.files.base:
    if hasattr(ctx.attr.base, "docker_layers"):
      # The base is the first layer in docker_layers if provided.
      return ctx.attr.base.docker_layers[0]["layer"]
    if len(ctx.files.base) != 1:
      fail("base attribute should be a single tar file.")
    return ctx.files.base[0]

def _serialize_dict(dict_value):
    return ",".join(["%s=%s" % (k, dict_value[k]) for k in dict_value])

def _image_config(ctx, layer_names):
  """Create the configuration for a new docker image."""
  config = ctx.new_file(ctx.label.name + ".config")

  label_file_dict = dict()
  for i in range(len(ctx.files.label_files)):
    fname = ctx.attr.label_file_strings[i]
    file = ctx.files.label_files[i]
    label_file_dict[fname] = file

  labels = dict()
  for l in ctx.attr.labels:
    fname = ctx.attr.labels[l]
    if fname[0] == '@':
      labels[l] = "@" + label_file_dict[fname[1:]].path
    else:
      labels[l] = fname

  args = [
      "--output=%s" % config.path,
      "--entrypoint=%s" % ",".join(ctx.attr.entrypoint),
      "--command=%s" % ",".join(ctx.attr.cmd),
      "--labels=%s" % _serialize_dict(labels),
      "--env=%s" % _serialize_dict(ctx.attr.env),
      "--ports=%s" % ",".join(ctx.attr.ports),
      "--volumes=%s" % ",".join(ctx.attr.volumes)
      ]
  if ctx.attr.user:
    args += ["--user=" + ctx.attr.user]
  if ctx.attr.workdir:
    args += ["--workdir=" + ctx.attr.workdir]

  inputs = layer_names
  args += ["--layer=@" + l.path for l in layer_names]

  if ctx.attr.label_files:
    inputs += ctx.files.label_files

  base = _get_base_artifact(ctx)
  if base:
    args += ["--base=%s" % base.path]
    inputs += [base]

  ctx.action(
      executable = ctx.executable.create_image_config,
      arguments = args,
      inputs = inputs,
      outputs = [config],
      use_default_shell_env=True,
      mnemonic = "ImageConfig")
  return config

def _metadata_action(ctx, layer, name, output):
  """Generate the action to create the JSON metadata for the layer."""
  rewrite_tool = ctx.executable.rewrite_tool

  label_file_dict = dict()
  for i in range(len(ctx.files.label_files)):
    fname = ctx.attr.label_file_strings[i]
    file = ctx.files.label_files[i]
    label_file_dict[fname] = file

  labels = dict()
  for l in ctx.attr.labels:
    fname = ctx.attr.labels[l]
    if fname[0] == '@':
      labels[l] = "@" + label_file_dict[fname[1:]].path
    else:
      labels[l] = fname

  args = [
      "--output=%s" % output.path,
      "--layer=%s" % layer.path,
      "--name=@%s" % name.path,
      "--entrypoint=%s" % ",".join(ctx.attr.entrypoint),
      "--command=%s" % ",".join(ctx.attr.cmd),
      "--labels=%s" % _serialize_dict(labels),
      "--env=%s" % _serialize_dict(ctx.attr.env),
      "--ports=%s" % ",".join(ctx.attr.ports),
      "--volumes=%s" % ",".join(ctx.attr.volumes)
      ]
  if ctx.attr.workdir:
    args += ["--workdir=" + ctx.attr.workdir]
  inputs = [layer, rewrite_tool, name]
  if ctx.attr.label_files:
    inputs += ctx.files.label_files
  base = _get_base_artifact(ctx)
  if base:
    args += ["--base=%s" % base.path]
    inputs += [base]
  if ctx.attr.user:
    args += ["--user=" + ctx.attr.user]

  ctx.action(
      executable = rewrite_tool,
      arguments = args,
      inputs = inputs,
      outputs = [output],
      use_default_shell_env=True,
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

def _create_image(ctx, layers, id, config, name, metadata):
  """Create the new image."""
  args = [
      "--output=" + ctx.outputs.layer.path,
      "--id=@" + id.path,
      "--config=" + config.path,
      ]

  args += ["--layer=@%s=%s" % (l["name"].path, l["layer"].path) for l in layers]
  inputs = [id, config] + [l["name"] for l in layers] + [l["layer"] for l in layers]

  if name:
    args += ["--legacy_id=@" + name.path]
    inputs += [name]

  if metadata:
    args += ["--metadata=" + metadata.path]
    inputs += [metadata]

  # If we have been provided a base image, add it.
  if ctx.attr.base and not hasattr(ctx.attr.base, "docker_layers"):
    legacy_base = _get_base_artifact(ctx)
    if legacy_base:
      args += ["--legacy_base=%s" % legacy_base.path]
      inputs += [legacy_base]

  base = _get_base_artifact(ctx)
  if base:
    args += ["--base=%s" % base.path]
    inputs += [base]
  ctx.action(
      executable = ctx.executable.create_image,
      arguments = args,
      inputs = inputs,
      outputs = [ctx.outputs.layer],
      mnemonic = "CreateImage",
      )

def _assemble_image(ctx, layers, name):
  """Create the full image from the list of layers."""
  layers = [l["layer"] for l in layers]
  args = [
      "--output=" + ctx.outputs.out.path,
      "--id=@" + name.path,
      "--repository=" + _repository_name(ctx),
      "--name=" + ctx.label.name
      ] + ["--layer=" + l.path for l in layers]
  inputs = [name] + layers
  ctx.action(
      executable = ctx.executable.join_layers,
      arguments = args,
      inputs = inputs,
      outputs = [ctx.outputs.out],
      mnemonic = "JoinLayers"
      )

def _repository_name(ctx):
  """Compute the repository name for the current rule."""
  return "%s/%s" % (ctx.attr.repository, ctx.label.package.replace("/", "_"))

def reverse(lst):
  result = []
  for el in lst:
    result = [el] + result
  return result

def _get_runfile_path(ctx, f):
  """Return the runfiles relative path of f."""
  if ctx.workspace_name:
    return ctx.workspace_name + "/" + f.short_path
  else:
    return f.short_path

def _docker_build_impl(ctx):
  """Implementation for the docker_build rule."""
  layer = _build_layer(ctx)
  layer_sha = _sha256(ctx, layer)

  config = _image_config(ctx, [layer_sha])
  id = _sha256(ctx, config)

  name = _compute_layer_name(ctx, layer)
  metadata = _metadata(ctx, layer, name)

  # creating a partial image so only pass the layers that belong to it
  image_layer = {"layer": layer, "name": layer_sha}
  _create_image(ctx, [image_layer], id, config, name, metadata)

  # Compute the layers transitive provider.
  # It includes the current layers, and, if they exists the layer from
  # base docker_build rules. We do not extract the list of layer in
  # a base tarball as they probably do not respect the convention on
  # layer naming that our rules use.
  layers =  [
      {"layer": ctx.outputs.layer, "id": id, "name": name}
      ] + getattr(ctx.attr.base, "docker_layers", [])
  # Generate the incremental load statement
  ctx.template_action(
      template = ctx.file.incremental_load_template,
      substitutions = {
        "%{load_statements}": "\n".join([
            "incr_load '%s' '%s' '%s'" % (_get_runfile_path(ctx, l["name"]),
                                          _get_runfile_path(ctx, l["id"]),
                                          _get_runfile_path(ctx, l["layer"]))
            # The last layer is the first in the list of layers.
            # We reverse to load the layer from the parent to the child.
            for l in reverse(layers)]),
        "%{repository}": _repository_name(ctx),
        "%{tag}": ctx.label.name,
        },
      output = ctx.outputs.executable,
      executable = True)
  _assemble_image(ctx, reverse(layers), name)
  runfiles = ctx.runfiles(
      files = [l["name"] for l in layers] +
              [l["id"] for l in layers] +
              [l["layer"] for l in layers])
  return struct(runfiles = runfiles,
                files = set([ctx.outputs.layer]),
                docker_layers = layers)

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
        "user": attr.string(),
        "env": attr.string_dict(),
        "labels": attr.string_dict(),
        "ports": attr.string_list(),  # Skylark doesn't support int_list...
        "volumes": attr.string_list(),
        "workdir": attr.string(),
        "repository": attr.string(default="bazel"),
        # Implicit dependencies.
        "label_files": attr.label_list(
            allow_files=True),
        "label_file_strings": attr.string_list(),
        "build_layer": attr.label(
            default=Label("//tools/build_defs/pkg:build_tar"),
            cfg=HOST_CFG,
            executable=True,
            allow_files=True),
        "create_image": attr.label(
            default=Label("//tools/build_defs/docker:create_image"),
            cfg=HOST_CFG,
            executable=True,
            allow_files=True),
        "incremental_load_template": attr.label(
            default=Label("//tools/build_defs/docker:incremental_load_template"),
            single_file=True,
            allow_files=True),
        "join_layers": attr.label(
            default=Label("//tools/build_defs/docker:join_layers"),
            cfg=HOST_CFG,
            executable=True,
            allow_files=True),
        "rewrite_tool": attr.label(
            default=Label("//tools/build_defs/docker:rewrite_json"),
            cfg=HOST_CFG,
            executable=True,
            allow_files=True),
        "create_image_config": attr.label(
            default=Label("//tools/build_defs/docker:create_image_config"),
            cfg=HOST_CFG,
            executable=True,
            allow_files=True),
        "sha256": attr.label(
            default=Label("//tools/build_defs/docker:sha256"),
            cfg=HOST_CFG,
            executable=True,
            allow_files=True)
    },
    outputs = {
        "out": "%{name}.tar",
        "layer": "%{name}-layer.tar",
    },
    executable = True)

# Produces a new docker image tarball compatible with 'docker load', which
# is a single additional layer atop 'base'.  The goal is to have relatively
# complete support for building docker image, from the Dockerfile spec.
#
# For more information see the 'Config' section of the image specification:
# https://github.com/opencontainers/image-spec/blob/v0.2.0/serialization.md
#
# Only 'name' is required. All other fields have sane defaults.
#
#   docker_build(
#      name="...",
#      visibility="...",
#
#      # The base layers on top of which to overlay this layer,
#      # equivalent to FROM.
#      base="//another/build:rule",
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
#      # https://docs.docker.com/reference/builder/#user
#      # NOTE: the normal directive affects subsequent RUN, CMD,
#      # and ENTRYPOINT
#      user="...",
#
#      # https://docs.docker.com/reference/builder/#volume
#      volumes=[...],
#
#      # https://docs.docker.com/reference/builder/#workdir
#      # NOTE: the normal directive affects subsequent RUN, CMD,
#      # ENTRYPOINT, ADD, and COPY, but this attribute only affects
#      # the entry point.
#      workdir="...",
#
#      # https://docs.docker.com/reference/builder/#env
#      env = {
#         "var1": "val1",
#         "var2": "val2",
#         ...
#         "varN": "valN",
#      },
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
    {image-config-sha256}.json
    ...
    manifest.json
    repositories
    top     # an implementation detail of our rules, not consumed by Docker.
  This rule appends a single new layer to the tarball of this form provided
  via the 'base' parameter.

  The images produced by this rule are always named 'bazel/tmp:latest' when
  loaded (an internal detail).  The expectation is that the images produced
  by these rules will be uploaded using the 'docker_push' rule below.

  Args:
    **kwargs: See above.
  """
  if "cmd" in kwargs:
    kwargs["cmd"] = _validate_command("cmd", kwargs["cmd"])
  for reserved in ["label_files", "label_file_strings"]:
    if reserved in kwargs:
      fail("reserved for internal use by docker_build macro", attr=reserved)
  if "labels" in kwargs:
    files = sorted(set([v[1:] for v in kwargs["labels"].values() if v[0] == '@']))
    kwargs["label_files"] = files
    kwargs["label_file_strings"] = files
  if "entrypoint" in kwargs:
    kwargs["entrypoint"] = _validate_command("entrypoint", kwargs["entrypoint"])
  docker_build_(**kwargs)
