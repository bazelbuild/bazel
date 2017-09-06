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

package com.google.devtools.build.lib.analysis;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import java.util.List;

/** Helper class to manage rules' use of platforms. */
public class PlatformSemantics {

  public static final String TARGET_PLATFORMS_ATTR = ":target_platforms";
  public static final String EXECUTION_PLATFORM_ATTR = ":execution_platform";
  public static final String TOOLCHAINS_ATTR = "$toolchains";

  /** Implementation for the :target_platform attribute. */
  public static final Attribute.LateBoundLabelList<BuildConfiguration> TARGET_PLATFORM =
      new Attribute.LateBoundLabelList<BuildConfiguration>(PlatformConfiguration.class) {
        @Override
        public List<Label> resolve(
            Rule rule, AttributeMap attributes, BuildConfiguration configuration) {
          // rule may be null for tests
          if (rule == null || rule.getRuleClassObject().getRequiredToolchains().isEmpty()) {
            return ImmutableList.of();
          }
          return configuration.getFragment(PlatformConfiguration.class).getTargetPlatforms();
        }
      };

  /** Implementation for the :execution_platform attribute. */
  public static final Attribute.LateBoundLabel<BuildConfiguration> EXECUTION_PLATFORM =
      new Attribute.LateBoundLabel<BuildConfiguration>(PlatformConfiguration.class) {
        @Override
        public Label resolve(Rule rule, AttributeMap attributes, BuildConfiguration configuration) {
          // rule may be null for tests
          if (rule == null || rule.getRuleClassObject().getRequiredToolchains().isEmpty()) {
            return null;
          }
          return configuration.getFragment(PlatformConfiguration.class).getExecutionPlatform();
        }
      };

  public static RuleClass.Builder platformAttributes(RuleClass.Builder builder) {
    return builder
        .add(
            attr(TARGET_PLATFORMS_ATTR, LABEL_LIST)
                .value(TARGET_PLATFORM)
                .nonconfigurable("Used in toolchain resolution"))
        .add(
            attr(EXECUTION_PLATFORM_ATTR, LABEL)
                .value(EXECUTION_PLATFORM)
                .nonconfigurable("Used in toolchain resolution"))
        .add(
            attr(TOOLCHAINS_ATTR, LABEL_LIST)
                .nonconfigurable("Used in toolchain resolution")
                .value(ImmutableList.of()));
  }
}
