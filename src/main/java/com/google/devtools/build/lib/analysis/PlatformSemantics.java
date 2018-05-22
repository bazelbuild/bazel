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
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;

/** Helper class to manage rules' use of platforms. */
public class PlatformSemantics {

  public static final String TOOLCHAINS_ATTR = "$toolchains";
  public static final String EXEC_COMPATIBLE_WITH_ATTR = "exec_compatible_with";

  public static RuleClass.Builder platformAttributes(RuleClass.Builder builder) {
    return builder
        .add(
            attr(TOOLCHAINS_ATTR, LABEL_LIST)
                .nonconfigurable("Used in toolchain resolution")
                .value(ImmutableList.of()));
  }

  /**
   * Return the target-specific execution platform constraints, based on the rule definition and any
   * constraints added by the target.
   */
  public static ImmutableSet<Label> getExecutionPlatformConstraints(Rule rule) {
    NonconfigurableAttributeMapper mapper = NonconfigurableAttributeMapper.of(rule);
    ImmutableSet.Builder<Label> execConstraintLabels = new ImmutableSet.Builder<>();

    if (rule.getRuleClassObject().executionPlatformConstraintsAllowed().allowsRule()) {
      // Add any default rule-level constraints.
      execConstraintLabels.addAll(rule.getRuleClassObject().getExecutionPlatformConstraints());
    }

    // Add any target-level constraints, if allowed.
    if (rule.getRuleClassObject().executionPlatformConstraintsAllowed().allowsTarget()
        && mapper.has(PlatformSemantics.EXEC_COMPATIBLE_WITH_ATTR)) {
      execConstraintLabels.addAll(
          mapper.get(PlatformSemantics.EXEC_COMPATIBLE_WITH_ATTR, BuildType.LABEL_LIST));
    }

    return execConstraintLabels.build();
  }
}
