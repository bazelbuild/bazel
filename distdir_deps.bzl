# Copyright 2020 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""List the distribution dependencies we need to build Bazel."""

DIST_DEPS = {
    # This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/cpp/cc_configure.WORKSPACE.
    # This must be kept in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
    "rules_cc": {
        "archive": "b1c40e1de81913a3c40e5948f78719c28152486d.zip",
        "sha256": "d0c573b94a6ef20ef6ff20154a23d0efcb409fb0e1ff0979cec318dfe42f0cdd",
        "strip_prefix": "ules_cc-b1c40e1de81913a3c40e5948f78719c28152486d",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_cc/archive/b1c40e1de81913a3c40e5948f78719c28152486d.zip",
            "https://github.com/bazelbuild/rules_cc/archive/b1c40e1de81913a3c40e5948f78719c28152486d.zip",
        ],
        "need_in_test_WORKSPACE": True,
    },
    "rules_pkg": {
        "archive": "rules_pkg-0.2.5.tar.gz",
        "sha256": "352c090cc3d3f9a6b4e676cf42a6047c16824959b438895a76c2989c6d7c246a",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.2.5/rules_pkg-0.2.5.tar.gz",
            "https://github.com/bazelbuild/rules_pkg/releases/download/0.2.5/rules_pkg-0.2.5.tar.gz",
        ],
    },
}

def _gen_workspace_stanza_impl(ctx):
    if ctx.attr.template and (ctx.attr.preamble or ctx.attr.postamble):
        fail("Can not use template with either preamble or postamble")

    repo_clause = """
maybe(
    http_archive,
    "{repo}",
    sha256 = "{sha256}",
    strip_prefix = {strip_prefix},
    urls = "{urls}",
)
"""
    repo_stanzas = {}
    for repo in ctx.attr.repos:
        info = DIST_DEPS[repo]
        strip_prefix = info.get("strip_prefix")
        if strip_prefix:
            strip_prefix = "\"%s\"" % strip_prefix
        else:
            strip_prefix = "None"

        repo_stanzas["{%s}" % repo] = repo_clause.format(
            repo = repo,
            archive = info["archive"],
            sha256 = str(info["sha256"]),
            strip_prefix = strip_prefix,
            urls = info["urls"],
        )

    if ctx.attr.template:
        print(repo_stanzas)
        ctx.actions.expand_template(
            output = ctx.outputs.out,
            template = ctx.file.template,
            substitutions = repo_stanzas,
        )
    else:
        content = "\n".join([l.strip() for l in ctx.attr.preamble.strip().split("\n")])
        content += "\n"
        content += "".join(repo_stanzas.values())
        content += "\n"
        content += "\n".join([l.strip() for l in ctx.attr.postamble.strip().split("\n")])
        content += "\n"
        ctx.actions.write(ctx.outputs.out, content)

    return [DefaultInfo(files = depset([ctx.outputs.out]))]

gen_workspace_stanza = rule(
    implementation = _gen_workspace_stanza_impl,
    attrs = {
        "repos": attr.string_list(doc = "Set of repos to inlcude"),
        "out": attr.output(mandatory = True),
        "preamble": attr.string(doc = "Preamble."),
        "postamble": attr.string(doc = "setup rules to follow repos."),
        "template": attr.label(
            doc = "Template WORKSPACE file. May not be used with preable or postamble." +
                  "Repo stanzas can be include with the syntax '{repo name}'.",
            allow_single_file = True,
            mandatory = False,
        ),
    },
)
