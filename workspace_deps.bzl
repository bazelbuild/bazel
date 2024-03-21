# Copyright 2020 The Bazel Authors. All rights reserved.
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

"""WORKSPACE default repository definitions."""

WORKSPACE_REPOS = {
    # Used in src/main/java/com/google/devtools/build/lib/bazel/rules/cpp/cc_configure.WORKSPACE.
    # Used in src/main/java/com/google/devtools/build/lib/bazel/rules/java/jdk.WORKSPACE.
    # Used in src/test/java/com/google/devtools/build/lib/blackbox/framework/blackbox.WORKSAPCE
    "rules_cc": {
        "archive": "rules_cc-0.0.9.tar.gz",
        "sha256": "2037875b9a4456dce4a79d112a8ae885bbc4aad968e6587dca6e64f3a0900cdf",
        "urls": ["https://github.com/bazelbuild/rules_cc/releases/download/0.0.9/rules_cc-0.0.9.tar.gz"],
        "strip_prefix": "rules_cc-0.0.9",
    },
    "rules_java": {
        "archive": "rules_java-7.5.0.tar.gz",
        "sha256": "4da3761f6855ad916568e2bfe86213ba6d2637f56b8360538a7fb6125abf6518",
        "urls": ["https://github.com/bazelbuild/rules_java/releases/download/7.5.0/rules_java-7.5.0.tar.gz"],
    },
    # Used in src/test/java/com/google/devtools/build/lib/blackbox/framework/blackbox.WORKSAPCE
    "rules_proto": {
        "archive": "5.3.0-21.7.tar.gz",
        "sha256": "dc3fb206a2cb3441b485eb1e423165b231235a1ea9b031b4433cf7bc1fa460dd",
        "strip_prefix": "rules_proto-5.3.0-21.7",
        "urls": ["https://github.com/bazelbuild/rules_proto/archive/refs/tags/5.3.0-21.7.tar.gz"],
    },
    "bazel_skylib": {
        "archive": "bazel-skylib-1.5.0.tar.gz",
        "sha256": "cd55a062e763b9349921f0f5db8c3933288dc8ba4f76dd9416aac68acee3cb94",
        "urls": ["https://github.com/bazelbuild/bazel-skylib/releases/download/1.5.0/bazel-skylib-1.5.0.tar.gz"],
    },
    "rules_license": {
        "archive": "rules_license-0.0.7.tar.gz",
        "sha256": "4531deccb913639c30e5c7512a054d5d875698daeb75d8cf90f284375fe7c360",
        "urls": ["https://github.com/bazelbuild/rules_license/releases/download/0.0.7/rules_license-0.0.7.tar.gz"],
    },
    "rules_python": {
        "archive": "rules_python-0.24.0.tar.gz",
        "sha256": "0a8003b044294d7840ac7d9d73eef05d6ceb682d7516781a4ec62eeb34702578",
        "strip_prefix": "rules_python-0.24.0",
        "urls": ["https://github.com/bazelbuild/rules_python/releases/download/0.24.0/rules_python-0.24.0.tar.gz"],
    },
    "rules_pkg": {
        "archive": "rules_pkg-0.9.1.tar.gz",
        "sha256": "8f9ee2dc10c1ae514ee599a8b42ed99fa262b757058f65ad3c384289ff70c4b8",
        "urls": ["https://github.com/bazelbuild/rules_pkg/releases/download/0.9.1/rules_pkg-0.9.1.tar.gz"],
    },
    "rules_testing": {
        "archive": "rules_testing-v0.0.4.tar.gz",
        "sha256": "4e21f9aa7996944ce91431f27bca374bff56e680acfe497276074d56bc5d9af2",
        "strip_prefix": "rules_testing-0.0.4",
        "urls": ["https://github.com/bazelbuild/rules_testing/releases/download/v0.0.4/rules_testing-v0.0.4.tar.gz"],
    },
    "remote_coverage_tools": {
        "archive": "coverage_output_generator-v2.6.zip",
        "sha256": "7006375f6756819b7013ca875eab70a541cf7d89142d9c511ed78ea4fefa38af",
        "urls": ["https://mirror.bazel.build/bazel_coverage_output_generator/releases/coverage_output_generator-v2.6.zip"],
    },
}

def _gen_workspace_stanza_impl(ctx):
    if ctx.attr.template and (ctx.attr.preamble or ctx.attr.postamble):
        fail("Can not use template with either preamble or postamble")
    if ctx.attr.use_maybe and ctx.attr.repo_clause:
        fail("Can not use use_maybe with repo_clause")

    if ctx.attr.use_maybe:
        repo_clause = """
maybe(
    http_archive,
    name = "{repo}",
    sha256 = "{sha256}",
    strip_prefix = {strip_prefix},
    urls = {urls},
)
"""
    elif ctx.attr.repo_clause:
        repo_clause = ctx.attr.repo_clause
    else:
        repo_clause = """
http_archive(
    name = "{repo}",
    sha256 = "{sha256}",
    strip_prefix = {strip_prefix},
    urls = {urls},
)
"""

    repo_stanzas = {}
    for repo in ctx.attr.repos:
        info = WORKSPACE_REPOS[repo]
        strip_prefix = info.get("strip_prefix")
        if strip_prefix:
            strip_prefix = "\"%s\"" % strip_prefix
        else:
            strip_prefix = "None"

        repo_stanzas["{%s}" % repo] = repo_clause.format(
            repo = repo,
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
    attrs = {
        "repos": attr.string_list(doc = "Set of repos to include."),
        "out": attr.output(mandatory = True),
        "preamble": attr.string(doc = "Preamble."),
        "postamble": attr.string(doc = "Set of rules to follow repos."),
        "template": attr.label(
            doc = "Template WORKSPACE file. May not be used with preamble or postamble." +
                  "Repo stanzas can be included using the syntax '{repo name}'.",
            allow_single_file = True,
            mandatory = False,
        ),
        "use_maybe": attr.bool(doc = "Use maybe() invocation instead of http_archive."),
        "repo_clause": attr.string(doc = "Use a custom clause for each repository."),
    },
    doc = "Use specifications from WORKSPACE_REPOS to generate WORKSPACE http_archive stanzas or to" +
          "drop them into a template.",
    implementation = _gen_workspace_stanza_impl,
)
