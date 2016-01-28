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

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.rules.repository.WorkspaceBaseRule;
import com.google.devtools.build.lib.rules.repository.WorkspaceConfiguredTargetFactory;
import com.google.devtools.build.lib.syntax.Type;

/**
 * Rule definition for the maven_jar rule.
 */
public class MavenJarRule implements RuleDefinition {

  public static final String NAME = "maven_jar";

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        /* <!-- #BLAZE_RULE(maven_jar).ATTRIBUTE(artifact) -->
        A description of a Maven artifact.

        <p>These descriptions are of the form &lt;groupId&gt:&lt;artifactId&gt;:&lt;version&gt;,
        see <a href="#maven_jar_examples">the documentation below</a> for an example.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("artifact", Type.STRING))
        .add(attr("artifact_id", Type.STRING).undocumented("deprecated"))
        .add(attr("group_id", Type.STRING).undocumented("deprecated"))
        .add(attr("version", Type.STRING).undocumented("deprecated"))
        /* <!-- #BLAZE_RULE(maven_jar).ATTRIBUTE(repository) -->
        A URL for a Maven repository to fetch the jar from.

        <p>Either this or <code>server</code> can be specified. Defaults to Maven Central
         ("central.maven.org").</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("repository", Type.STRING))
        /* <!-- #BLAZE_RULE(maven_jar).ATTRIBUTE(server) -->
        A maven_server to use for this artifact.

        <p>Either this or <code>repository</code> can be specified.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("server", Type.STRING))
        /* <!-- #BLAZE_RULE(maven_jar).ATTRIBUTE(sha1) -->
         A SHA-1 hash of the desired jar.

         <p>If the downloaded jar does not match this hash, Bazel will error out.</p>
         <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("sha1", Type.STRING))
        .setWorkspaceOnly()
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name(MavenJarRule.NAME)
        .type(RuleClassType.WORKSPACE)
        .ancestors(WorkspaceBaseRule.class)
        .factoryClass(WorkspaceConfiguredTargetFactory.class)
        .build();
  }
}
/*<!-- #BLAZE_RULE (NAME = maven_jar, TYPE = OTHER, FAMILY = Workspace)[GENERIC_RULE] -->

<p>Downloads a jar from Maven and makes it available to be used as a Java dependency.</p>

<h4 id="maven_jar_examples">Examples</h4>

Suppose that the current repostory contains a java_library target that needs to depend on Guava.
Using Maven, this dependency would be defined in the pom.xml file as:

<pre>
<dependency>
    <groupId>com.google.guava</groupId>
    <artifactId>guava</artifactId>
    <version>18.0</version>
</dependency>
</pre>

With Bazel, add the following lines to the WORKSPACE file:

<pre>
maven_jar(
    name = "guava",
    artifact = "com.google.guava:guava:18.0",
    sha1 = "cce0823396aa693798f8882e64213b1772032b09",
)
</pre>

<p>Targets can specify <code>@guava//jar</code> as a dependency to depend on this jar.</p>

<!-- #END_BLAZE_RULE -->*/
