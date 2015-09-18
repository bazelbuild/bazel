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

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
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
        A path to a settings.xml file.
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

${ATTRIBUTE_SIGNATURE}

<p>How to access a Maven repository.</p>

${ATTRIBUTE_DEFINITION}

<p>This is a combination of a &lt;repository&gt; definition from a pom.xml file and a
&lt;server&lt; definition from a settings.xml file.</p>

<!-- #END_BLAZE_RULE -->*/
