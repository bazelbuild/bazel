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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.platform.ConstraintSettingInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.skyframe.SkyFunction;
import javax.annotation.Nullable;

/**
 * Defines a category of constraint that can be fulfilled by a constraint_value rule in a platform
 * definition.
 */
public class ConstraintSetting implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {

    Label constraintSetting = ruleContext.getLabel();
    Label defaultConstraintValue =
        ruleContext
            .attributes()
            .get(ConstraintSettingRule.DEFAULT_CONSTRAINT_VALUE_ATTR, BuildType.NODEP_LABEL);

    validateDefaultConstraintValue(ruleContext, constraintSetting, defaultConstraintValue);

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .addProvider(FileProvider.class, FileProvider.EMPTY)
        .addProvider(FilesToRunProvider.class, FilesToRunProvider.EMPTY)
        .addNativeDeclaredProvider(
            ConstraintSettingInfo.create(constraintSetting, defaultConstraintValue))
        .build();
  }

  private void validateDefaultConstraintValue(
      RuleContext ruleContext, Label constraintSetting, @Nullable Label defaultConstraintValue)
      throws RuleErrorException, InterruptedException {
    if (defaultConstraintValue == null) {
      return;
    }

    // Make sure the default value is in the same package.
    if (!constraintSetting
        .getPackageIdentifier()
        .equals(defaultConstraintValue.getPackageIdentifier())) {
      throw ruleContext.throwWithAttributeError(
          ConstraintSettingRule.DEFAULT_CONSTRAINT_VALUE_ATTR,
          "The default constraint value must be defined in the same package "
              + "as the constraint setting itself.");
    }

    // Verify that the target actually exists, even though we cannot have a direct dependency
    // because it will cause a cycle.
    SkyFunction.Environment env = ruleContext.getAnalysisEnvironment().getSkyframeEnv();
    PackageValue packageNode =
        (PackageValue) env.getValue(constraintSetting.getPackageIdentifier());
    Preconditions.checkNotNull(
        packageNode,
        "Package '%s' is the package for the current target, and so must have already been loaded.",
        defaultConstraintValue.getPackageIdentifier());
    Package pkg = packageNode.getPackage();
    try {
      pkg.getTarget(defaultConstraintValue.getName());
    } catch (NoSuchTargetException e) {
      throw ruleContext.throwWithAttributeError(
          ConstraintSettingRule.DEFAULT_CONSTRAINT_VALUE_ATTR,
          "The default constraint value '" + defaultConstraintValue + "' does not exist");
    }
  }
}
