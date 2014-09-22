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

"""Example of a genrule reimplementation using skylark.

Example of use:

load("rules/genrule", "genrule_skylark")

genrule_skylark(
    name = "world",
    outs = ["hi"],
    cmd = "touch $(@)",
)
"""


def resolve_command(ctx, command, resolved_srcs, files_to_build):
  variables = {"SRCS": files.join_exec_paths(" ", resolved_srcs),
               "OUTS": files.join_exec_paths(" ", files_to_build)}
  if len(resolved_srcs) == 1:
    variables["<"] = list(resolved_srcs)[0].path
  if len(files_to_build) == 1:
    variables["@"] = list(files_to_build)[0].path
  return ctx.expand_make_variables("cmd", command, variables)


def create(ctx):
  resolved_srcs = nset("STABLE_ORDER")
  files_to_build = nset("STABLE_ORDER", ctx.outputs("outs"))
  if not files_to_build:
    ctx.error("outs", "genrules without outputs don't make sense")

  if ctx.attr.executable and len(files_to_build) > 1:
    ctx.error("executable",
          "if genrules produce executables, they are allowed only one output. "
          + "If you need the executable=1 argument, then you should split this "
          + "genrule into genrules producing single outputs")

  label_dict = {}
  for dep in ctx.targets("srcs", "TARGET"):
    files = provider(dep, "FileProvider").files_to_build
    resolved_srcs += files
    label_dict[dep.label] = files

  command_helper = ctx.command_helper(
      tools=ctx.targets("tools", "HOST", "FilesToRunProvider"),
      label_dict=label_dict)

  # TODO(bazel_team): Implement heuristic label expansion
  command = command_helper.resolve_command_and_expand_labels(False, False)

  # TODO(bazel_team): Improve resolve_command method
  command = resolve_command(ctx,
                            command,
                            nset("STABLE_ORDER", resolved_srcs),
                            files_to_build)

  message = ctx.attr.message or "Executing genrule"

  # TODO(bazel_team): Do something with requires-FEATURE tags.
  env = ctx.configuration.default_shell_env

  resolved_srcs += command_helper.resolved_tools
  command_line_srcs = []
  # TODO(bazel-team): this method still has side effects which we rely on
  argv = command_helper.build_command_line(
      command, command_line_srcs, ".genrule_script.sh")
  resolved_srcs += command_line_srcs

  # TODO(bazel_team): Maybe implement stamp attribute?

  ctx.action(inputs=resolved_srcs,
             outputs=files_to_build,
             env=env,
             command=argv,
             progress_message="%s %s" % (message, ctx),
             mnemonic="Genrule")

  # TODO(bazel_team): Implement and add Instrumented files collector?

  # Executable has to be specified explicitly
  if ctx.attr.executable:
    return struct(files_to_build=files_to_build,
                  runfiles=ctx.runfiles(data=[files_to_build]),
                  executable=list(files_to_build)[0])
  else:
    return struct(files_to_build=files_to_build,
                  runfiles=ctx.runfiles(data=[files_to_build]))


genrule_skylark = rule(implementation=create,
     # TODO(bazel_team): Do we need these for now?
     # .setDependentTargetConfiguration(PARENT)
     # .setOutputToGenfiles()
     attr={
         "srcs": attr.label_list(flags=["DIRECT_COMPILE_TIME_INPUT"],
             file_types=ANY_FILE),
         "tools": attr.label_list(cfg=HOST_CFG, file_types=ANY_FILE),
         "outs": attr.output_list(mandatory=True),
         "cmd": attr.string(mandatory=True),
         "message": attr.string(),
         "output_licenses": attr.license(),
         "executable": attr.bool(default=False),
         },
    )
