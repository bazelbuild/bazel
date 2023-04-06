# Copyright 2021 The Bazel Authors. All rights reserved.
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

"""Common util functions for java_* rules"""

load(":common/java/java_semantics.bzl", "semantics")

def create_single_jar(ctx, output, *input_depsets):
    """Register action for the output jar.

    Args:
      ctx: (RuleContext) Used to register the action.
      output: (Artifact) Output file of the action.
      *input_depsets: (list[depset[Artifact]]) Input files of the action.

    Returns:
      (File) Output file which was used for registering the action.
    """
    toolchain = semantics.find_java_toolchain(ctx)
    args = ctx.actions.args()
    args.set_param_file_format("shell").use_param_file("@%s", use_always = True)
    args.add("--output", output)
    args.add_all(
        [
            "--compression",
            "--normalize",
            "--exclude_build_data",
            "--warn_duplicate_resources",
        ],
    )
    all_inputs = depset(transitive = input_depsets)
    args.add_all("--sources", all_inputs)

    ctx.actions.run(
        mnemonic = "JavaSingleJar",
        progress_message = "Building singlejar jar %s" % output.short_path,
        executable = toolchain.single_jar,
        toolchain = semantics.JAVA_TOOLCHAIN_TYPE,
        inputs = all_inputs,
        tools = [toolchain.single_jar],
        outputs = [output],
        arguments = [args],
    )
    return output

# TODO(hvd): use skylib shell.quote()
def shell_quote(s):
    return "'" + s.replace("'", "'\\''") + "'"
