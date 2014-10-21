# Copyright 2014 Google Inc. All rights reserved.
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

"""Maprule implementation using skylark.

Attribute documentation: see at the bottom where the attributes are defined.

Example of use:

load("foo/bar/baz/maprule", "maprule")

maprule(
    name = "extract_js_deps",
    srcs = [":header_gen"],
    outs = {
        "prov": "$(src)_provides.js",
        "req": "$(src)_requires.js",
    },
    cmd = " ; ".join([
        "(cat $(SRCS)",
        "echo '// Generated from $(src)'",
        "grep 'goog.require' $(src) || echo \"// nothing required\"",
        ") > $(req)",
        "(cat $(SRCS)",
        "echo '// Generated from $(src)'",
        "grep 'goog.provide' $(src) || echo \"// nothing provided\"",
        ") > $(prov)",
    ]),
    foreach_srcs = [":js_files"],
)

filegroup(
    name = "js_files",
    srcs = glob(["*.js"]) + ["//foo/bar:js_files"],
)

genrule(
    name = "header_gen",
    outs = ["header.txt"],
    cmd = "echo '// Copyright 2014 Google Inc.' > $@",
)
"""


def create_common_make_vars(srcs):
  """Returns a dict with Make Variables common to all per-source actions."""
  return {"SRCS": cmd_helper.join_paths(" ", srcs)}


def add_make_var(ctx, var_dict, name, value, attr_name):
  """Helper function to add new Make variables."""
  if name in var_dict:
    fail("duplicate Make Variable \"%s\"" % name, attr=attr_name)
  return {name: value}


def resolve_command(ctx, common_vars, command, srcs, foreach_src,
    templates_dict):
  """Resolves Make Variables in the command string."""
  variables = {}
  variables += common_vars
  # Add the user-defined Make Variables for the foreach_src's outputs.
  for makevar in templates_dict:
    variables += add_make_var(ctx, variables, makevar,
        templates_dict[makevar].path, "outs")
  # Add the $(@) makevar if there's only one output per source.
  if len(templates_dict) == 1:
    variables += add_make_var(ctx, variables, '@',
        templates_dict.values()[0].path, "outs")
  # Add the $(src) makevar for foreach_src.
  variables += add_make_var(ctx, variables, "src", foreach_src.path,
      "foreach_srcs")
  return ctx.expand_make_variables("cmd", command, variables)


def get_files_to_build(targets):
  """Returns all files built by targets in 'targets'."""
  result = set()
  for dep in targets:
    result += dep.files
  return result


def get_outs_templates(ctx):
  """Returns a dict of the output templates + checks them and reports errors."""
  outs_templ = ctx.attr.outs
  if not outs_templ:
    fail("must not be empty", attr="outs")

  result = {}
  values = {}  # set of values; "in" operator only works on dicts
  for key in outs_templ:
    value = outs_templ[key]
    error_prefix = "in template declaration (\"%s\": \"%s\") - " % (key, value)
    if len(value.split("$(src)")) != 2:
      fail(error_prefix + "must contain the placeholder $(src) exactly once",
           attr="outs")
    if value == '$(src)':
      fail(error_prefix + "must be more than just a mere placeholder",
           attr="outs")
    if key in result:
      fail(error_prefix + "duplicate Make Variable name \"%s\"" % key,
           attr="outs")
    if value in values:
      fail(error_prefix + "duplicate output name \"%s\"" % value,
           attr="outs")
    result[key] = value
    values[value] = None  # dict is used as a set
  return result


def fill_outs_templates(ctx, foreach_srcs, templates):
  """Replaces $(src) in output templates with the current source's path."""
  result = {}
  for src in foreach_srcs:
    outs = {}
    for key in templates:
      templ = templates[key].replace("$(src)", src.short_path)
      output = ctx.new_file(ctx.configuration.genfiles_dir,
          ctx.label.name + ".outputs/" + templ)
      outs[key] = output
    result[src] = outs
  return result


def create(ctx):
  # Resolve targets in "srcs" and "foreach_srcs" to files they output.
  common_srcs = get_files_to_build(ctx.targets.srcs)
  foreach_srcs = get_files_to_build(ctx.targets.foreach_srcs)
  if not foreach_srcs:
    fail("must not be empty", attr="foreach_srcs")

  # Create a dict for the output templates.
  # Key: Make Variable name of the output; Value: template
  templates = get_outs_templates(ctx)

  # Create the outputs for the foreach_srcs.
  foreach_src_outs_dict = fill_outs_templates(ctx, foreach_srcs, templates)

  command_helper = ctx.command_helper(
      tools=ctx.targets.tools,
      label_dict={})  # TODO(bazel-team): labels we pass here are not used

  command = command_helper.resolve_command_and_expand_labels(False, False)
  common_srcs += command_helper.resolved_tools

  makevars = create_common_make_vars(set(common_srcs))
  env = ctx.configuration.default_shell_env
  message = ctx.attr.message or "Executing maprule"

  foreach_src_outs = []
  for src in foreach_srcs:
    outs_dict = foreach_src_outs_dict[src]
    foreach_src_outs += outs_dict.values()
    cmd = resolve_command(ctx, makevars, command, common_srcs, src, outs_dict)

    command_line_srcs = []
    # TODO(bazel-team): this method still has side effects which we rely on
    argv = command_helper.build_command_line(cmd, command_line_srcs,
        ".genrule_script.sh")
    common_srcs += command_line_srcs

    ctx.action(inputs=list(common_srcs + [src]), outputs=outs_dict.values(),
        env=env, command=argv, progress_message="%s %s" % (message, ctx),
        mnemonic="Maprule")

  files_to_build = set(foreach_src_outs)
  return struct(files=files_to_build,
      data_runfiles=ctx.runfiles(transitive_files=files_to_build))


maprule = rule(implementation=create,
     # TODO(bazel_team): Do we need these for now?
     # .setDependentTargetConfiguration(PARENT)
     # .setOutputToGenfiles()
     attrs={
         # List of labels; optional.
         # Defines the set of sources that are available to all actions created
         # by this rule.
         #
         # This attribute would better be called "common_srcs", but $(location)
         # expansion only works for srcs, deps, data and tools.
         "srcs": attr.label_list(flags=["DIRECT_COMPILE_TIME_INPUT"],
             allow_files=True),

         # List of labels; required.
         # Defines the set of sources that will be processed one by one in
         # parallel to produce the templated outputs. For each action created
         # by this rule only one of these sources will be provided.
         #
         # This attribute would better be called "srcs", but that we need for
         # the common srcs in order to make $(location) expansion work.
         "foreach_srcs": attr.label_list(flags=["DIRECT_COMPILE_TIME_INPUT"],
             allow_files=True, mandatory=True),

         # List of labels; optional.
         # Tools used by the command in "cmd". Similar to genrule.tools
         "tools": attr.label_list(cfg=HOST_CFG, allow_files=True),

         # Dict of output templates; required.
         # Defines the templates for the outputs generated for each file in the
         # "foreach_srcs". Each key in this dict defines a Make Variable which
         # stands for the output template, the value under the key. The Make
         # Variable can be used in the "cmd" and will resolve to the name of
         # that particular output. The template must contain the placeholder
         # $(src) exactly once which will be replaced with the path of the
         # particular input file. If this dictionary contains only one entry,
         # the $(@) Make Variable can also be used to identify the output.
         "outs": attr.string_dict(mandatory=True),

         # String; required.
         # The shell command to execute for each file of the "foreach_srcs" that
         # produces the outputs. Similar to genrule.cmd
         "cmd": attr.string(mandatory=True),

         # String; optional.
         # The progress message to display when the actions are being executed.
         "message": attr.string(),

         # List of strings; optional.
         # See the common attribute definitions in the Build Encyclopedia.
         "output_licenses": attr.license(),
         },
    )
