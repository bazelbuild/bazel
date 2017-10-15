// Copyright 2015 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.bazel.rules.workspace;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;
import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.repository.WorkspaceBaseRule;
import com.google.devtools.build.lib.rules.repository.WorkspaceConfiguredTargetFactory;

/**
 * Rule definition for the new_git_repository rule.
 */
public class NewGitRepositoryRule implements RuleDefinition {
  public static final String NAME = "new_git_repository";

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        /* <!-- #BLAZE_RULE(new_git_repository).ATTRIBUTE(remote) -->
        The URI of the remote Git repository.

        <p>This must be a HTTP URL. There is currently no support for authentication.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("remote", STRING).mandatory())
        /* <!-- #BLAZE_RULE(git_repository).ATTRIBUTE(commit) -->
        The commit hash to check out in the repository.

        <p>Note that one of either <code>commit</code> or <code>tag</code> must be defined.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("commit", STRING))
        /* <!-- #BLAZE_RULE(git_repository).ATTRIBUTE(tag) -->
        The Git tag to check out in the repository.

        <p>Note that one of either <code>commit</code> or <code>tag</code> must be defined.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("tag", STRING))
        /* <!-- #BLAZE_RULE(new_git_repository).ATTRIBUTE(build_file) -->
        The file to use as the BUILD file for this repository.

        <p>Either build_file or build_file_content must be specified.</p>

        <p>This attribute is a label relative to the main workspace. The file does not need to be
        named BUILD, but can be. (Something like BUILD.new-repo-name may work well for
        distinguishing it from the repository's actual BUILD files.)</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("build_file", STRING))
        /* <!-- #BLAZE_RULE(new_git_repository).ATTRIBUTE(build_file_content) -->
        The content for the BUILD file for this repository.

        <p>Either build_file or build_file_content must be specified.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("build_file_content", STRING))
        /* <!-- #BLAZE_RULE(new_http_archive).ATTRIBUTE(workspace_file) -->
        The file to use as the WORKSPACE file for this repository.

         <p>Either workspace_file or workspace_file_content can be specified, but not both.</p>

         <p>This attribute is a label relative to the main workspace. The file does not need to be
        named WORKSPACE, but can be. (Something like WORKSPACE.new-repo-name may work well for
        distinguishing it from the repository's actual WORKSPACE files.)</p>
         <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("workspace_file", STRING))
        /* <!-- #BLAZE_RULE(new_http_archive).ATTRIBUTE(workspace_file_content) -->
        The content for the WORKSPACE file for this repository.

         <p>Either workspace_file or workspace_file_content can be specified, but not both.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("workspace_file_content", STRING))
        /* <!-- #BLAZE_RULE(new_git_repository).ATTRIBUTE(init_submodules) -->
        Whether to clone submodules in the repository.

        <p>Currently, only cloning the top-level submodules is supported</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("init_submodules", BOOLEAN).value(false))
        /* <!-- #BLAZE_RULE(new_git_repository).ATTRIBUTE(sha256) -->
        The expected SHA-256 hash of the file downloaded. Specifying this forces the repository to
        be downloaded as a tarball. Currently, this is only supported for public GitHub
        repositories.

        <p>This must match the SHA-256 hash of the file downloaded. <em>It is a security risk to
        omit the SHA-256 as remote files can change.</em> At best omitting this field will make
        your build non-hermetic. It is optional to make development easier but should be set
        before shipping.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("sha256", STRING))
        .setWorkspaceOnly()
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name(NewGitRepositoryRule.NAME)
        .type(RuleClass.Builder.RuleClassType.WORKSPACE)
        .ancestors(WorkspaceBaseRule.class)
        .factoryClass(WorkspaceConfiguredTargetFactory.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = new_git_repository, TYPE = OTHER, FAMILY = Workspace)[GENERIC_RULE] -->

<em><p>Warning: this rule has several limitations. We recommend using
<a href="#new_http_archive"><code>new_http_archive</code></a> instead for more robust and
secure performance.</p>

<p><code>new_git_repository</code> has several issues:

<ul>
<li>Security - <code>new_http_archive</code> allows a sha256 to be specified, which ensures that
the downloaded code is exactly what was expected.
<li>Reliability - <code>new_http_archive</code> allows the user to specify multiple URLs to attempt
downloads from. Most services have downtime occasionally, so specifying multiple remotes decreases
the chances of being unable to download a dependency.
<li>Speed - <code>new_http_archive</code> multiplexes downloads to get the fastest possible rate.
You can also generally download a tarball that is a "shallow clone" of the repository, which
decreases the size of the download.
<li>Library issues - This implementation uses jGit, which we've discovered
<a href="https://github.com/bazelbuild/bazel/issues/2802">several issues</a> with. It also lacks
support for some authentication types you might use with your system git.
</ul>

<p>Many git repository hosts serve tarballs of the repository, so depend on those if possible.
For GitHub, this takes the form:

<pre>
new_http_archive(
    name = "<name>",
    urls = ["https://github.com/&lt;user&gt;/&lt;repo&gt;/archive/&lt;commit or tag&gt;.tar.gz"],
    build_file = "<build file>",
)
</pre>

If you are using a private repository, prefer the
<a href="https://github.com/bazelbuild/bazel/blob/master/tools/build_defs/repo/git.bzl">Skylark git
repository rules</a>, which will use your system's git install (instead of jGit). These rules
are built into Bazel and have the same API as the native rules.</p></em>

<p>Clones a Git repository, checks out the specified tag, or commit, and makes its targets
available for binding.</p>

<h4 id="git_repository_examples">Examples</h4>

<p>Suppose the current repository contains the source code for a chat program, rooted at the
  directory <i>~/chat-app</i>. It needs to depend on an SSL library which is available in the
  remote Git repository <i>http://example.com/openssl/openssl.git</i>. The chat app depends
  on version 1.0.2 of the SSL library, which is tagged by the v1.0.2 Git tag.<p>

<p>This Git repository contains the following directory structure:</p>

<pre class="code">
src/
  openssl.cc
  openssl.h
</pre>

<p>Targets in the <i>~/chat-app</i> repository can depend on this target if the following lines are
  added to <i>~/chat-app/WORKSPACE</i>:</p>

<pre class="code">
new_git_repository(
    name = "my_ssl",
    remote = "http://example.com/openssl/openssl.git",
    tag = "v1.0.2",
    build_file_content = """
cc_library(
    name = "openssl-lib",
    srcs = ["src/openssl.cc"],
    hdrs = ["src/openssl.h"],
)""",
)
</pre>

<p>Then targets would specify <code>@my_ssl//:openssl-lib</code> as a dependency.</p>

<!-- #END_BLAZE_RULE -->*/
