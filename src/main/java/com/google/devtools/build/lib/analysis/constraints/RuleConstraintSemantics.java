// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.constraints;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.packages.PackageSpecification;
import com.google.devtools.build.lib.packages.PackageGroupsRuleVisibility;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.analysis.IncompatiblePlatformProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.ConfigConditions;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMapBuilder;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.packages.PackageSpecification.PackageGroupContents;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.build.lib.analysis.test.TestActionBuilder;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.RuleConfiguredTargetValue;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;

public class RuleConstraintSemantics {
  public static RuleConfiguredTargetValue createIncompatibleRuleConfiguredTarget(
      Target target,
      BuildConfigurationValue configuration,
      ConfigConditions configConditions,
      IncompatiblePlatformProvider incompatiblePlatformProvider,
      String ruleClassString,
      NestedSetBuilder<Package> transitivePackagesForPackageRootResolution
      ) {
    NestedSet<Artifact> filesToBuild = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    FileProvider fileProvider = new FileProvider(filesToBuild);
    FilesToRunProvider filesToRunProvider = new FilesToRunProvider(filesToBuild, null, null);

    TransitiveInfoProviderMapBuilder providerBuilder =
        new TransitiveInfoProviderMapBuilder()
            .put(incompatiblePlatformProvider)
            .add(RunfilesProvider.simple(Runfiles.EMPTY))
            .add(fileProvider)
            .add(filesToRunProvider);
    if (configuration.hasFragment(TestConfiguration.class)) {
      // Create a dummy TestProvider instance so that other parts of the code base stay happy. Even
      // though this test will never execute, some code still expects the provider.
      TestProvider.TestParams testParams = TestActionBuilder.createEmptyTestParams();
      providerBuilder.put(TestProvider.class, new TestProvider(testParams));
    }

    RuleConfiguredTarget configuredTarget = new RuleConfiguredTarget(
        target.getLabel(),
        configuration.getKey(),
        convertVisibility(target),
        providerBuilder.build(),
        configConditions.asProviders(),
        ruleClassString);
    return new RuleConfiguredTargetValue(
        configuredTarget,
        transitivePackagesForPackageRootResolution == null
            ? null
            : transitivePackagesForPackageRootResolution.build());
  }

  // Taken from ConfiguredTargetFactory.convertVisibility().
  // TODO(phil): Figure out how to de-duplicate somehow.
  private static NestedSet<PackageGroupContents> convertVisibility(Target target) {
    RuleVisibility ruleVisibility = target.getVisibility();
    if (ruleVisibility instanceof ConstantRuleVisibility) {
      return ((ConstantRuleVisibility) ruleVisibility).isPubliclyVisible()
          ? NestedSetBuilder.create(
              Order.STABLE_ORDER,
              PackageGroupContents.create(ImmutableList.of(PackageSpecification.everything())))
          : NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    } else if (ruleVisibility instanceof PackageGroupsRuleVisibility) {
      throw new IllegalStateException("unsupported PackageGroupsRuleVisibility");
    } else {
      throw new IllegalStateException("unknown visibility");
    }
  }

}
