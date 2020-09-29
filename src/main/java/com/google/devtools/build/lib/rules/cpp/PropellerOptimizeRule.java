// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;

/** {@code propeller_optimize} rule class. */
public final class PropellerOptimizeRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(CppConfiguration.class)
        /* <!-- #BLAZE_RULE(propeller_optimize).ATTRIBUTE(profile) -->
        Label of the profile passed to the various compile actions.  This file has
        the .txt extension.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("cc_profile", LABEL)
                .allowedFileTypes(FileTypeSet.of(FileType.of(".txt")))
                .singleArtifact())
        /* <!-- #BLAZE_RULE(propeller_optimize).ATTRIBUTE(ld_profile) -->
        Label of the profile passed to the link action.  This file has
        the .txt extension.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("ld_profile", LABEL)
                .allowedFileTypes(FileTypeSet.of(FileType.of(".txt")))
                .singleArtifact())
        .advertiseProvider(PropellerOptimizeProvider.class)
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("propeller_optimize")
        .ancestors(BaseRuleClasses.BaseRule.class)
        .factoryClass(PropellerOptimize.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = propeller_optimize, TYPE = LIBRARY, FAMILY = C / C++) -->

<p>Represents a Propeller optimization profile in the workspace.
Example:</p>

<pre class="code">
propeller_optimize(
    name = "layout",
    profile = "//path:profile.txt",
    ld_profile = "//path:ld_profile.txt"
)

</pre>
<!-- #END_BLAZE_RULE -->*/
