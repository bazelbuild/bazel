// Copyright 2018 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;

/** {@code fdo_prefetch_hints} rule class. */
public final class FdoPrefetchHintsRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(CppConfiguration.class)
        /* <!-- #BLAZE_RULE(fdo_prefetch_hints).ATTRIBUTE(profile) -->
        Label of the hints profile. The hints file has the .afdo extension
        The label can also point to an fdo_absolute_path_profile rule.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("profile", LABEL)
                .allowedFileTypes(
                    FileTypeSet.of(
                        FileType.of(".afdo")))
                .singleArtifact())
        /* <!-- #BLAZE_RULE(fdo_profile).ATTRIBUTE(absolute_path_profile) -->
        Absolute path to the FDO profile. The FDO file may only have the .afdo extension.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("absolute_path_profile", Type.STRING))
        .advertiseProvider(FdoPrefetchHintsProvider.class)
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("fdo_prefetch_hints")
        .ancestors(BaseRuleClasses.NativeBuildRule.class)
        .factoryClass(FdoPrefetchHints.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = fdo_prefetch_hints, TYPE = LIBRARY, FAMILY = C / C++) -->

<p>Represents an FDO prefetch hints profile that is either in the workspace or at a specified
absolute path.
Examples:</p>

<pre class="code">
fdo_prefetch_hints(
    name = "hints",
    profile = "//path/to/hints:profile.afdo",
)

fdo_profile(
  name = "hints_abs",
  absolute_path_profile = "/absolute/path/profile.afdo",
)
</pre>
<!-- #END_BLAZE_RULE -->*/
