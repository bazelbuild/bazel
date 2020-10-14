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

package com.google.devtools.build.lib.rules.repository;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.STRING;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;

/**
 * Rule definition for the new_repository rule.
 */
public class NewLocalRepositoryRule implements RuleDefinition {
  public static final String NAME = "new_local_repository";

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        /* <!-- #BLAZE_RULE(new_local_repository).ATTRIBUTE(path) -->
        A path on the local filesystem.

        <p>This must be an absolute path to an existing file or a directory.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("path", STRING).mandatory())
        /* <!-- #BLAZE_RULE(new_local_repository).ATTRIBUTE(build_file) -->
        A file to use as a BUILD file for this directory.

        <p>Either build_file or build_file_content must be specified.</p>

        <p>This attribute is a label relative to the main workspace. The file does not need to be
        named BUILD, but can be. (Something like BUILD.new-repo-name may work well for
        distinguishing it from the repository's actual BUILD files.)</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("build_file", STRING))
        /* <!-- #BLAZE_RULE(new_local_repository).ATTRIBUTE(build_file_content) -->
        The content for the BUILD file for this repository.

        <p>Either build_file or build_file_content must be specified.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("build_file_content", STRING))
        /* <!-- #BLAZE_RULE(new_local_repository).ATTRIBUTE(workspace_file) -->
        The file to use as the WORKSPACE file for this repository.

         <p>Either workspace_file or workspace_file_content can be specified, but not both.</p>

         <p>This attribute is a label relative to the main workspace. The file does not need to be
        named WORKSPACE, but can be. (Something like WORKSPACE.new-repo-name may work well for
        distinguishing it from the repository's actual WORKSPACE files.)</p>
         <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("workspace_file", STRING))
        /* <!-- #BLAZE_RULE(new_local_repository).ATTRIBUTE(workspace_file_content) -->
        The content for the WORKSPACE file for this repository.

         <p>Either workspace_file or workspace_file_content can be specified, but not both.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("workspace_file_content", STRING))
        .setWorkspaceOnly()
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name(NewLocalRepositoryRule.NAME)
        .type(RuleClassType.WORKSPACE)
        .ancestors(WorkspaceBaseRule.class)
        .factoryClass(WorkspaceConfiguredTargetFactory.class)
        .build();
  }
}
/*<!-- #BLAZE_RULE (NAME = new_local_repository, FAMILY = Workspace)[GENERIC_RULE] -->

<p>Allows a local directory to be turned into a Bazel repository. This means that the current
  repository can define and use targets from anywhere on the filesystem.</p>

<p>This rule creates a Bazel repository by creating a WORKSPACE file and subdirectory containing
symlinks to the BUILD file and path given.  The build file should create targets relative to the
<code>path</code>. For directories that already contain a WORKSPACE file and a BUILD file, the
<a href="#local_repository"><code>local_repository</code></a> rule can be used.

<h4 id="new_local_repository_examples">Examples</h4>

<p>Suppose the current repository is a chat client, rooted at the directory <i>~/chat-app</i>. It
  would like to use an SSL library which is defined in a different directory: <i>~/ssl</i>.</p>

<p>The user can add a dependency by creating a BUILD file for the SSL library
(~/chat-app/BUILD.my-ssl) containing:

<pre class="code">
java_library(
    name = "openssl",
    srcs = glob(['*.java'])
    visibility = ["//visibility:public"],
)
</pre>

<p>Then they can add the following lines to <i>~/chat-app/WORKSPACE</i>:</p>

<pre class="code">
new_local_repository(
    name = "my-ssl",
    path = "/home/user/ssl",
    build_file = "BUILD.my-ssl",
)
</pre>

<p>This will create a <code>@my-ssl</code> repository that symlinks to <i>/home/user/ssl</i>.
Targets can depend on this library by adding <code>@my-ssl//:openssl</code> to a target's
dependencies.</p>

<p>You can also use <code>new_local_repository</code> to include single files, not just
directories. For example, suppose you had a jar file at /home/username/Downloads/piano.jar. You
could add just that file to your build by adding the following to your WORKSPACE file:

<pre class="code">
new_local_repository(
    name = "piano",
    path = "/home/username/Downloads/piano.jar",
    build_file = "BUILD.piano",
)
</pre>

<p>And creating the following BUILD.piano file:</p>

<pre class="code">
java_import(
    name = "play-music",
    jars = ["piano.jar"],
    visibility = ["//visibility:public"],
)
</pre>

Then targets can depend on <code>@piano//:play-music</code> to use piano.jar.

<!-- #END_BLAZE_RULE -->*/
