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
package com.google.devtools.build.lib.rules.java;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.Types;
import com.google.devtools.build.lib.vfs.PathFragment;

/** A helper class to create configured targets for Java rules. */
public class JavaCommon {

  private JavaCommon() {}

  public static ImmutableList<String> getConstraints(RuleContext ruleContext) {
    return ruleContext.getRule().isAttrDefined("constraints", Types.STRING_LIST)
        ? ImmutableList.copyOf(ruleContext.attributes().get("constraints", Types.STRING_LIST))
        : ImmutableList.of();
  }

  /** Returns the per-package configured runfiles. */
  public static NestedSet<Artifact> computePerPackageData(
      RuleContext ruleContext, JavaToolchainProvider toolchain) throws RuleErrorException {
    // Do not use streams here as they create excessive garbage.
    NestedSetBuilder<Artifact> data = NestedSetBuilder.naiveLinkOrder();
    for (JavaPackageConfigurationProvider provider : toolchain.packageConfiguration()) {
      if (provider.matches(ruleContext.getLabel())) {
        data.addTransitive(provider.data());
      }
    }
    return data.build();
  }

  public static PathFragment getHostJavaExecutable(JavaRuntimeInfo javaRuntime)
      throws RuleErrorException {
    return javaRuntime.javaBinaryExecPathFragment();
  }

  /**
   * Returns true if and only if this target has the neverlink attribute set to 1, or false if the
   * neverlink attribute does not exist (for example, on *_binary targets)
   *
   * @return the value of the neverlink attribute.
   */
  public static final boolean isNeverLink(RuleContext ruleContext) {
    return ruleContext.getRule().isAttrDefined("neverlink", Type.BOOLEAN)
        && ruleContext.attributes().get("neverlink", Type.BOOLEAN);
  }
}
