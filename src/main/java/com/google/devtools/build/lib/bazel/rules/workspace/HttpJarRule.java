// Copyright 2014 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.rules.repository.WorkspaceBaseRule;
import com.google.devtools.build.lib.rules.repository.WorkspaceConfiguredTargetFactory;

/**
 * Rule definition for the http_jar rule.
 */
public class HttpJarRule implements RuleDefinition {

  public static final String NAME = "http_jar";

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        /* <!-- #BLAZE_RULE(http_jar).ATTRIBUTE(url) -->
        A URL to an archive file containing a Bazel repository.

        <p>This must be an http or https URL that ends with .jar. Redirections are followed.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("url", STRING).mandatory())
        /* <!-- #BLAZE_RULE(http_jar).ATTRIBUTE(sha256) -->
        The expected SHA-256 of the file downloaded.

        <p>This must match the SHA-256 of the file downloaded. <em>It is a security risk to
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
        .name(HttpJarRule.NAME)
        .type(RuleClassType.WORKSPACE)
        .ancestors(WorkspaceBaseRule.class)
        .factoryClass(WorkspaceConfiguredTargetFactory.class)
        .build();
  }
}
/*<!-- #BLAZE_RULE (NAME = http_jar, TYPE = OTHER, FAMILY = Workspace)[GENERIC_RULE] -->

<p>Downloads a jar from a URL and makes it available to be used as a Java dependency.</p>

<p>Downloaded files must have a .jar extension.</p>

<h4 id="http_jar_examples">Examples</h4>

<p>Suppose the current repository contains the source code for a chat program, rooted at the
  directory <i>~/chat-app</i>. It needs to depend on an SSL library which is available from
  <i>http://example.com/openssl-0.2.jar</i>.</p>

<p>Targets in the <i>~/chat-app</i> repository can depend on this target if the following lines are
  added to <i>~/chat-app/WORKSPACE</i>:</p>

<pre class="code">
http_jar(
    name = "my-ssl",
    url = "http://example.com/openssl-0.2.jar",
    sha256 = "03a58ac630e59778f328af4bcc4acb4f80208ed4",
)
</pre>

<p>Targets would specify <code>@my-ssl//jar</code> as a dependency to depend on this jar.</p>

<!-- #END_BLAZE_RULE -->*/
