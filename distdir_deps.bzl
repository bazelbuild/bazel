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
    ########################################
    #
    # Runtime language dependencies
    #
    ########################################
    # Keep in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/cpp/cc_configure.WORKSPACE.
    # Keep in sync with src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
    # Note: This is not in sync with src/test/java/com/google/devtools/build/lib/blackbox/framework/BlackBoxTestEnvironment.java.
    #       Perhaps it should be.
    "rules_cc": {
        "archive": "b1c40e1de81913a3c40e5948f78719c28152486d.zip",
        "sha256": "d0c573b94a6ef20ef6ff20154a23d0efcb409fb0e1ff0979cec318dfe42f0cdd",
        "strip_prefix": "rules_cc-b1c40e1de81913a3c40e5948f78719c28152486d",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_cc/archive/b1c40e1de81913a3c40e5948f78719c28152486d.zip",
            "https://github.com/bazelbuild/rules_cc/archive/b1c40e1de81913a3c40e5948f78719c28152486d.zip",
        ],
        "need_in_test_WORKSPACE": True,
    },
    "rules_java": {
        "archive": "7cf3cefd652008d0a64a419c34c13bdca6c8f178.zip",
        "sha256": "bc81f1ba47ef5cc68ad32225c3d0e70b8c6f6077663835438da8d5733f917598",
        "strip_prefix": "rules_java-7cf3cefd652008d0a64a419c34c13bdca6c8f178",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_java/archive/7cf3cefd652008d0a64a419c34c13bdca6c8f178.zip",
            "https://github.com/bazelbuild/rules_java/archive/7cf3cefd652008d0a64a419c34c13bdca6c8f178.zip",
        ],
        "used_in": [
            "additional_distfiles",
            "test_WORKSPACE_files",
        ],
        "need_in_test_WORKSPACE": True,
    },

    ########################################
    #
    # Build time dependencies
    #
    ########################################
    "rules_pkg": {
        "archive": "rules_pkg-0.2.4.tar.gz",
        "sha256": "4ba8f4ab0ff85f2484287ab06c0d871dcb31cc54d439457d28fd4ae14b18450a",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.2.4/rules_pkg-0.2.4.tar.gz",
            "https://github.com/bazelbuild/rules_pkg/releases/download/0.2.4/rules_pkg-0.2.4.tar.gz",
        ],
    },
    # for Stardoc
    "io_bazel_rules_sass": {
        "archive": "1.25.0.zip",
        "sha256": "c78be58f5e0a29a04686b628cf54faaee0094322ae0ac99da5a8a8afca59a647",
        "strip_prefix": "rules_sass-1.25.0",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_sass/archive/1.25.0.zip",
            "https://github.com/bazelbuild/rules_sass/archive/1.25.0.zip",
        ],
    },
    # for Stardoc
    "build_bazel_rules_nodejs": {
        "archive": "rules_nodejs-2.2.2.tar.gz",
        "sha256": "f2194102720e662dbf193546585d705e645314319554c6ce7e47d8b59f459e9c",
        "urls": [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_nodejs/releases/download/2.2.2/rules_nodejs-2.2.2.tar.gz",
            "https://github.com/bazelbuild/rules_nodejs/releases/download/2.2.2/rules_nodejs-2.2.2.tar.gz",
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
    urls = {urls},
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
        ctx.actions.expand_template(
            output = ctx.outputs.out,
            template = ctx.file.template,
            substitutions = repo_stanzas,
        )
    else:
        content = "\n".join([p.strip() for p in ctx.attr.preamble.strip().split("\n")])
        content += "\n"
        content += "".join(repo_stanzas.values())
        content += "\n"
        content += "\n".join([p.strip() for p in ctx.attr.postamble.strip().split("\n")])
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
