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
"""Implementation for Bazel Python executable."""

# @unused
def _create_executable_bazel(
        ctx,
        *,
        executable,
        main_py,
        imports,
        is_test,
        runtime_details,
        cc_details,
        native_deps_details,
        runfiles_details):
    # TODO: Implement the zip file and Windows logic. The logic here is rather
    # convoluted. but basically, there are four files declared:
    #  * executable: either {name} or {name}.exe, name depending on platform
    #  * zip file: {name}.zip
    #  * windows launcher: {name}.exe; how it later invokes the stub is unclear
    #  * temp file: {name}.temp, the stub used by the zip file
    # See BazelPythonSemantics.createExecutable and postInitExecutable for
    # details.
    _ = imports, is_test, runtime_details, cc_details, native_deps_details, runfiles_details  # @unused
    ctx.actions.expand_template(
        template = template,
        output = executable,
        substitutions = {
            "%shebang": "<get from runtime_details>",
            "%main": main_py.short_path,
            "%python_binary%": "<get path to interpreter>",
            "%coverage_tool%": "<coverage tool path>",
            "%imports%": "<build imports path colon-list>",
            "%workspace_name%": ctx.workspace_name,
            "%is_zipfile%": "<get is zipfile flag>",
            "%import_all%": "<get import all flag>",
            "%target%": "<get name including repo>",
            # These keys are unused: %python_version_from_config%,
            # %python_version_from_attr%, %python_version_specified_explicitly%
        },
        is_executable = True,
    )
