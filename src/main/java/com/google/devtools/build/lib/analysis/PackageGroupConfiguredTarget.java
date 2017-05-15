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

package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.PackageSpecification;
import com.google.devtools.build.lib.util.Preconditions;

/**
 * Dummy ConfiguredTarget for package groups. Contains no functionality, since
 * package groups are not really first-class Targets.
 */
public final class PackageGroupConfiguredTarget extends AbstractConfiguredTarget
    implements PackageSpecificationProvider {
  private final NestedSet<PackageSpecification> packageSpecifications;

  PackageGroupConfiguredTarget(TargetContext targetContext, PackageGroup packageGroup) {
    super(targetContext);
    Preconditions.checkArgument(targetContext.getConfiguration() == null);

    NestedSetBuilder<PackageSpecification> builder =
        NestedSetBuilder.stableOrder();
    for (Label label : packageGroup.getIncludes()) {
      TransitiveInfoCollection include = targetContext.maybeFindDirectPrerequisite(
          label, targetContext.getConfiguration());
      PackageSpecificationProvider provider = include == null ? null :
          include.getProvider(PackageSpecificationProvider.class);
      if (provider == null) {
        targetContext.getAnalysisEnvironment().getEventHandler().handle(Event.error(getTarget().getLocation(),
            String.format("label '%s' does not refer to a package group", label)));
        continue;
      }

      builder.addTransitive(provider.getPackageSpecifications());
    }

    builder.addAll(packageGroup.getPackageSpecifications());
    packageSpecifications = builder.build();
  }

  @Override
  public PackageGroup getTarget() {
    return (PackageGroup) super.getTarget();
  }

  @Override
  public NestedSet<PackageSpecification> getPackageSpecifications() {
    return packageSpecifications;
  }
}
