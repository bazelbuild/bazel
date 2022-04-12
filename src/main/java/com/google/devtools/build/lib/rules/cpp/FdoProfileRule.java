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
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.util.FileTypeSet;

/** {@code fdo_profile} rule class. */
public final class FdoProfileRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(CppConfiguration.class)
        /* <!-- #BLAZE_RULE(fdo_profile).ATTRIBUTE(profile) -->
        Label of the FDO profile. The FDO file can have one of the following extensions:
        .profraw for unindexed LLVM profile, .profdata for indexed LLVM profile, .zip that holds an
        LLVM profraw profile, .afdo for AutoFDO profile, .xfdo for XBinary profile.
        The label can also point to an fdo_absolute_path_profile rule.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("profile", LABEL)
                .allowedFileTypes(
                    FileTypeSet.of(
                        CppFileTypes.LLVM_PROFILE_RAW,
                        CppFileTypes.LLVM_PROFILE,
                        CppFileTypes.LLVM_PROFILE_ZIP,
                        CppFileTypes.GCC_AUTO_PROFILE,
                        CppFileTypes.XBINARY_PROFILE))
                .singleArtifact())
        /* <!-- #BLAZE_RULE(fdo_profile).ATTRIBUTE(absolute_path_profile) -->
        Absolute path to the FDO profile. The FDO file can have one of the following extensions:
        .profraw for unindexed LLVM profile, .profdata for indexed LLVM profile, .zip
        that holds an LLVM profraw profile, or .afdo for AutoFDO profile.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("absolute_path_profile", Type.STRING))
        /* <!-- #BLAZE_RULE(fdo_profile).ATTRIBUTE(proto_profile) -->
        Label of the protobuf profile.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("proto_profile", LABEL)
            .allowedFileTypes(FileTypeSet.ANY_FILE)
            .singleArtifact())
        .advertiseProvider(FdoProfileProvider.class)
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("fdo_profile")
        .ancestors(BaseRuleClasses.NativeBuildRule.class)
        .factoryClass(FdoProfile.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = fdo_profile, TYPE = LIBRARY, FAMILY = C / C++) -->

<p>Represents an FDO profile that is either in the workspace or at a specified absolute path.
Examples:</p>

<pre class="code">
fdo_profile(
    name = "fdo",
    profile = "//path/to/fdo:profile.zip",
)

fdo_profile(
  name = "fdo_abs",
  absolute_path_profile = "/absolute/path/profile.zip",
)
</pre>
<!-- #END_BLAZE_RULE -->*/
