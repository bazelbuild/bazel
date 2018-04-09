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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;

/** {@code fdo_profile} rule class. */
public final class FdoProfileRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        /* <!-- #BLAZE_RULE(fdo_profile).ATTRIBUTE(profile) -->
        Label of the FDO profile. The FDO file can have one of the following extensions:
        .profraw for unindexed LLVM profile, .profdata fo indexed LLVM profile, .zip
        that holds GCC gcda profile or LLVM profraw profile.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("profile", LABEL)
                .mandatory()
                .allowedFileTypes(
                    FileTypeSet.of(
                        CppFileTypes.LLVM_PROFILE_RAW,
                        CppFileTypes.LLVM_PROFILE,
                        FileType.of(".zip")))
                .singleArtifact())
        .advertiseProvider(FdoProfileProvider.class)
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("fdo_profile")
        .ancestors(BaseRuleClasses.BaseRule.class)
        .factoryClass(FdoProfile.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = fdo_profile, TYPE = LIBRARY, FAMILY = Cpp) -->

<p>Represents a checked-in FDO profile. Example:</p>

<pre class="code">
fdo_profile(
    name = "fdo",
    profile = "//path/to/fdo:profile.zip",
)
</pre>
<!-- #END_BLAZE_RULE -->*/
