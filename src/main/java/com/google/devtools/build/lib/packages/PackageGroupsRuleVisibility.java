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
package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.PackageSpecification.PackageGroupContents;
import java.util.Collection;
import java.util.List;

/** A rule visibility that allows visibility to a list of package groups. */
@Immutable
public class PackageGroupsRuleVisibility implements RuleVisibility {
  private final List<Label> packageGroups;
  private final PackageGroupContents directPackages;
  private final List<Label> declaredLabels;

  private PackageGroupsRuleVisibility(Label ruleLabel, List<Label> labels) {
    declaredLabels = ImmutableList.copyOf(labels);
    ImmutableList.Builder<PackageSpecification> directPackageBuilder = ImmutableList.builder();
    ImmutableList.Builder<Label> packageGroupBuilder = ImmutableList.builder();

    for (Label label : labels) {
      Label resolved = ruleLabel.resolveRepositoryRelative(label);
      PackageSpecification specification = PackageSpecification.fromLabel(resolved);
      if (specification != null) {
        directPackageBuilder.add(specification);
      } else {
        packageGroupBuilder.add(resolved);
      }
    }

    packageGroups = packageGroupBuilder.build();
    directPackages = PackageGroupContents.create(directPackageBuilder.build());
  }

  public Collection<Label> getPackageGroups() {
    return packageGroups;
  }

  public PackageGroupContents getDirectPackages() {
    return directPackages;
  }

  @Override
  public List<Label> getDependencyLabels() {
    return packageGroups;
  }

  @Override
  public List<Label> getDeclaredLabels() {
    return declaredLabels;
  }

  /**
   * Tries to parse a list of labels into a {@link PackageGroupsRuleVisibility}.
   *
   * @param labels the list of labels to parse
   * @return The resulting visibility object. A list of labels can always be
   * parsed into a PackageGroupsRuleVisibility.
   */
  public static PackageGroupsRuleVisibility tryParse(Label ruleLabel, List<Label> labels) {
    return new PackageGroupsRuleVisibility(ruleLabel, labels);
  }
}
