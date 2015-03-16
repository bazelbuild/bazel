// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.analysis.BlazeRule;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.Type;

/**
 * Rule definition for the maven_jar rule.
 */
@BlazeRule(name = MavenJarRule.NAME,
           type = RuleClassType.WORKSPACE,
           ancestors = { WorkspaceBaseRule.class },
           factoryClass = WorkspaceConfiguredTargetFactory.class)
public class MavenJarRule implements RuleDefinition {

  public static final String NAME = "maven_jar";

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        /* <!-- #BLAZE_RULE(maven_jar).ATTRIBUTE(artifact_id) -->
        The artifactId of the Maven dependency.
        ${SYNOPSIS}
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("artifact_id", Type.STRING).mandatory())
        /* <!-- #BLAZE_RULE(maven_jar).ATTRIBUTE(group_id) -->
        The groupId of the Maven dependency.
        ${SYNOPSIS}
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("group_id", Type.STRING).mandatory())
        /* <!-- #BLAZE_RULE(maven_jar).ATTRIBUTE(version) -->
        The version of the Maven dependency.
        ${SYNOPSIS}
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("version", Type.STRING).mandatory())
        /* <!-- #BLAZE_RULE(maven_jar).ATTRIBUTE(repositories) -->
        A list of repositories to use to attempt to fetch the jar.
        ${SYNOPSIS}

        <p>Defaults to Maven Central ("repo1.maven.org"). If repositories are specified, they will
          be checked in the order listed here (Maven Central will not be checked in this case,
          unless it is on the list).</p>

        <p><b>To be implemented: add a maven_repositories rule that allows a list of repositories
        to be labeled.</b></p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("repositories", Type.STRING_LIST))
        /* <!-- #BLAZE_RULE(maven_jar).ATTRIBUTE(exclusions) -->
        Transitive dependencies of this dependency that should not be downloaded.
        ${SYNOPSIS}

        <p>Defaults to None: Bazel will download all of the dependencies requested by the Maven
          dependency.  If exclusions are specified, they will not be downloaded.</p>

        <p>Exclusions are specified in the format "<group_id>:<artifact_id>", for example,
          "com.google.guava:guava".</p>

        <p><b>Not yet implemented.</b></p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("exclusions", Type.STRING_LIST))
        .setWorkspaceOnly()
        .build();
  }
}
/*<!-- #BLAZE_RULE (NAME = maven_jar, TYPE = OTHER, FAMILY = General)[GENERIC_RULE] -->

${ATTRIBUTE_SIGNATURE}

<p>Downloads a jar from Maven and makes it available to be used as a Java dependency.</p>

${ATTRIBUTE_DEFINITION}

<h4 id="http_jar_examples">Examples</h4>

Suppose that the current repostory contains a java_library target that needs to depend on Guava.
Using Maven, this dependency would be defined in the pom.xml file as:

<pre>
<dependency>
    <groupId>com.google.guava</groupId>
    <artifactId>guava</artifactId>
    <version>18.0</version>
</dependency>
</pre>

In Bazel, the following lines can be added to the WORKSPACE file:

<pre>
maven_jar(
    name = "guava",
    group_id = "com.google.guava",
    artifact_id = "guava",
    version = "18.0",
)

bind(
    name = "guava-jar",
    actual = "@guava//jar"
)
</pre>

Then the java_library can depend on <code>//external:guava-jar</code>.

<p>See <a href="#bind_examples">Bind</a> for how to use bound targets.</p>

<!-- #END_BLAZE_RULE -->*/
