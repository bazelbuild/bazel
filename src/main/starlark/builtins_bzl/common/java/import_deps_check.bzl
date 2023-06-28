# Copyright 2022 The Bazel Authors. All rights reserved.
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

"""Creates the import deps checker for java rules"""

load(":common/java/java_semantics.bzl", "semantics")

def import_deps_check(
        ctx,
        jars_to_check,
        declared_deps,
        transitive_deps,
        rule_class):
    """
    Creates actions that checks import deps for java rules.

    Args:
      ctx: (RuleContext) Used to register the actions.
      jars_to_check: (list[File])  A list of jars files to check.
      declared_deps: (list[File]) A list of direct dependencies.
      transitive_deps: (list[File]) A list of transitive dependencies.
      rule_class: (String) Rule class.

    Returns:
      (File) Output file of the created action.
    """
    java_toolchain = semantics.find_java_toolchain(ctx)
    deps_checker = java_toolchain.deps_checker()
    if deps_checker == None:
        return None

    jdeps_output = ctx.actions.declare_file("_%s/%s/jdeps.proto" % (rule_class, ctx.label.name))

    args = ctx.actions.args()
    args.add("-jar", deps_checker)
    args.add_all(jars_to_check, before_each = "--input")
    args.add_all(declared_deps, before_each = "--directdep")
    args.add_all(
        depset(order = "preorder", transitive = [declared_deps, transitive_deps]),
        before_each = "--classpath_entry",
    )
    args.add_all(java_toolchain.bootclasspath, before_each = "--bootclasspath_entry")
    args.add("--checking_mode=error")
    args.add("--jdeps_output", jdeps_output)
    args.add("--rule_label", ctx.label)

    inputs = depset(
        jars_to_check,
        transitive = [
            declared_deps,
            transitive_deps,
            java_toolchain.bootclasspath,
        ],
    )
    tools = [deps_checker, java_toolchain.java_runtime.files]

    ctx.actions.run(
        mnemonic = "ImportDepsChecker",
        progress_message = "Checking the completeness of the deps for %s" % jars_to_check,
        executable = java_toolchain.java_runtime.java_executable_exec_path,
        arguments = [args],
        inputs = inputs,
        outputs = [jdeps_output],
        tools = tools,
    )

    return jdeps_output
