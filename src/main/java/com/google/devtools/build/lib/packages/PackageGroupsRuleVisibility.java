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

  private PackageGroupsRuleVisibility(List<Label> labels) {
    declaredLabels = ImmutableList.copyOf(labels);
    ImmutableList.Builder<PackageSpecification> directPackageBuilder = ImmutableList.builder();
    ImmutableList.Builder<Label> packageGroupBuilder = ImmutableList.builder();

    for (Label label : labels) {
      PackageSpecification specification = PackageSpecification.fromLabel(label);
      if (specification != null) {
        directPackageBuilder.add(specification);
      } else {
        packageGroupBuilder.add(label);
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

  /** Creates a {@link PackageGroupsRuleVisibility} from a list of labels. */
  public static PackageGroupsRuleVisibility create(List<Label> labels) {
    return new PackageGroupsRuleVisibility(labels);
  }
}
