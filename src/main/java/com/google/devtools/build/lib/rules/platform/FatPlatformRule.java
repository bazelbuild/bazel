// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.platform;

import static com.google.devtools.build.lib.packages.Attribute.attr;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.FatPlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.util.FileTypeSet;

/** Rule definition for {@link FatPlatform}. */
public class FatPlatformRule implements RuleDefinition {
  public static final String RULE_NAME = "fat_platform";
  public static final String PLATFORMS_ATTR = "platforms";

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    /* <!-- #BLAZE_RULE(fat_platform).NAME -->
    <!-- #END_BLAZE_RULE.NAME --> */
    return builder
        .advertiseProvider(FatPlatformInfo.class, PlatformInfo.class)

        /* <!-- #BLAZE_RULE(fat_platform).ATTRIBUTE(platforms) -->
        TKTK
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr(PLATFORMS_ATTR, BuildType.LABEL_LIST)
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .mandatoryProviders(PlatformInfo.PROVIDER.id()))

        .build();
  }

  @Override
  public Metadata getMetadata() {
    return Metadata.builder()
        .name(RULE_NAME)
        .ancestors(PlatformBaseRule.class)
        .factoryClass(FatPlatform.class)
        .build();
  }
}
/*<!-- #BLAZE_RULE (NAME = fat_platform, FAMILY = Platform)[GENERIC_RULE] -->

TKTK
<!-- #END_BLAZE_RULE -->*/
