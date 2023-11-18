// Copyright 2023 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.util.FileTypeSet;

/** {@code memprof_profile} rule class. */
public final class MemProfProfileRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(CppConfiguration.class)
        /* <!-- #BLAZE_RULE(memprof_profile).ATTRIBUTE(profile) -->
        Label of the MEMPROF profile. The profile is expected to have
        either a .profdata extension (for an indexed/symbolized memprof
        profile), or a .zip extension for a zipfile containing a memprof.profdata
        file.
        The label can also point to an fdo_absolute_path_profile rule.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("profile", LABEL)
                .allowedFileTypes(
                    FileTypeSet.of(CppFileTypes.LLVM_PROFILE, CppFileTypes.LLVM_PROFILE_ZIP))
                .singleArtifact())
        /* <!-- #BLAZE_RULE(memprof_profile).ATTRIBUTE(absolute_path_profile) -->
        Absolute path to the MEMPROF profile. The file may only have a .profdata or
        .zip extension (where the zipfile must contain a memprof.profdata file).
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("absolute_path_profile", Type.STRING))
        .advertiseProvider(MemProfProfileProvider.class)
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("memprof_profile")
        .ancestors(BaseRuleClasses.NativeBuildRule.class)
        .factoryClass(MemProfProfile.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = memprof_profile, TYPE = LIBRARY, FAMILY = C / C++) -->

<p>Represents a MEMPROF profile that is either in the workspace or at a specified
absolute path.
Examples:</p>

<pre class="code">
memprof_profile(
    name = "memprof",
    profile = "//path/to/memprof:profile.afdo",
)

memprof_profile(
  name = "memprof_abs",
  absolute_path_profile = "/absolute/path/profile.afdo",
)
</pre>
<!-- #END_BLAZE_RULE -->*/
