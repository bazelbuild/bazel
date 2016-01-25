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
public class MavenServerRule implements RuleDefinition {

  public static final String NAME = "maven_server";

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        /* <!-- #BLAZE_RULE(maven_server).ATTRIBUTE(url) -->
        A URL for accessing the server.
        ${SYNOPSIS}

        <p>For example, Maven Central (which is the default and does not need to be defined) would
        be specified as <code>url = "http://central.maven.org/maven2/"</code>.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("url", Type.STRING))
        /* <!-- #BLAZE_RULE(maven_server).ATTRIBUTE(settings_file) -->
        A path to a settings.xml file.  Used for testing. If unspecified, this defaults to using
        <code>$M2_HOME/conf/settings.xml</code> for the global settings and
        <code>$HOME/.m2/settings.xml</code> for the user settings.
        ${SYNOPSIS}
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("settings_file", Type.STRING))
        .setWorkspaceOnly()
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name(MavenServerRule.NAME)
        .type(RuleClassType.WORKSPACE)
        .ancestors(WorkspaceBaseRule.class)
        .factoryClass(WorkspaceConfiguredTargetFactory.class)
        .build();
  }
}
/*<!-- #BLAZE_RULE (NAME = maven_server, TYPE = OTHER, FAMILY = Workspace)[GENERIC_RULE] -->

<p>How to access a Maven repository.</p>

<p>This is a combination of a &lt;repository&gt; definition from a pom.xml file and a
&lt;server&lt; definition from a settings.xml file.</p>

<h4>Using <code>maven_server</code></h4>

<p><code>maven_jar</code> rules can specify the name of a <code>maven_server</code> in their
<code>server</code> field. For example, suppose we have the following WORKSPACE file:</p>

<pre>
maven_jar(
    name = "junit",
    artifact = "junit:junit-dep:4.10",
    server = "my-server",
)

maven_server(
    name = "my-server",
    url = "http://intranet.mycorp.net",
)
</pre>

This specifies that junit should be downloaded from http://intranet.mycorp.net using the
authentication information found in ~/.m2/settings.xml (specifically, the settings
for the server with the id <code>my-server</code>).

<h4>Specifying a default server</h4>

<p>If you create a <code>maven_server</code> with the <code>name</code> "default" it will be
used for any <code>maven_jar</code> that does not specify a <code>server</code> nor
<code>repository</code>. If there is no <code>maven_server</code> named default, the
default will be fetching from Maven Central with no authentication enabled.</p>

<!-- #END_BLAZE_RULE -->*/
