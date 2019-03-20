# Copyright 2019 The Bazel Authors. All rights reserved.
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
def _test_rule(ctx):
    out = ctx.actions.declare_file("out.txt")
    files = ctx.attr.module_source.files
    found = False
    for file_ in files:
        if file_.basename == "package.json":
            compare_version(ctx.actions, file_, out, ctx.attr.version)
            found = True
            break
    if not found:
        fail("Not found package.json")
    return [DefaultInfo(files = depset([out]))]

test_rule = rule(
    implementation = _test_rule,
    attrs = {
        "module_source": attr.label(),
        "version": attr.string(),
    },
)

def compare_version(action_factory, file_, out, expected_version):
    action_factory.run_shell(
        mnemonic = "getVersion",
        inputs = [file_],
        outputs = [out],
        command = """result=$(cat ./{file} | grep '"version": "{expected}"' || exit 1) \
&& echo $result > ./{out}""".format(
            file = file_.path,
            out = out.path,
            expected = expected_version,
        ),
    )
