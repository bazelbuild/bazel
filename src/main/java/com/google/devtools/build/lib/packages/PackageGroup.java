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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.packages.PackageSpecification.PackageGroupContents;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading.Code;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import net.starlark.java.syntax.Location;

/**
 * This class represents a package group BUILD target. It has a name, a list of {@link
 * PackageSpecification}s, a list of {@link Label}s of other package groups this one includes, and
 * can be asked if a specific package is included in it.
 */
public class PackageGroup implements Target {
  private boolean containsErrors;
  private final Label label;
  private final Location location;
  private final Package containingPackage;
  private final PackageGroupContents packageSpecifications;
  private final List<Label> includes;

  public PackageGroup(
      Label label,
      Package pkg,
      Collection<String> packageSpecifications,
      Collection<Label> includes,
      EventHandler eventHandler,
      Location location) {
    this.label = label;
    this.location = location;
    this.containingPackage = pkg;
    this.includes = ImmutableList.copyOf(includes);

    // TODO(bazel-team): Consider refactoring so constructor takes a PackageGroupContents.
    ImmutableList.Builder<PackageSpecification> packagesBuilder = ImmutableList.builder();
    for (String packageSpecification : packageSpecifications) {
      PackageSpecification specification = null;
      try {
        specification =
            PackageSpecification.fromString(label.getRepository(), packageSpecification);
      } catch (PackageSpecification.InvalidPackageSpecificationException e) {
        containsErrors = true;
        eventHandler.handle(
            Package.error(location, e.getMessage(), Code.INVALID_PACKAGE_SPECIFICATION));
      }

      if (specification != null) {
        packagesBuilder.add(specification);
      }
    }
    this.packageSpecifications = PackageGroupContents.create(packagesBuilder.build());
  }

  public boolean containsErrors() {
    return containsErrors;
  }

  public PackageGroupContents getPackageSpecifications() {
    return packageSpecifications;
  }

  public boolean contains(Package pkg) {
    return packageSpecifications.containsPackage(pkg.getPackageIdentifier());
  }

  public List<Label> getIncludes() {
    return includes;
  }

  public List<String> getContainedPackages() {
    return packageSpecifications.containedPackages().collect(toImmutableList());
  }

  @Override
  public Rule getAssociatedRule() {
    return null;
  }

  @Override
  public Set<DistributionType> getDistributions() {
    return Collections.emptySet();
  }

  @Override
  public Label getLabel() {
    return label;
  }

  @Override public String getName() {
    return label.getName();
  }

  @Override
  public License getLicense() {
    return License.NO_LICENSE;
  }

  @Override
  public Package getPackage() {
    return containingPackage;
  }

  @Override
  public String getTargetKind() {
    return targetKind();
  }

  @Override
  public Location getLocation() {
    return location;
  }

  @Override
  public String toString() {
   return targetKind() + " " + getLabel();
  }

  @Override
  public RuleVisibility getVisibility() {
    // Package groups are always public to avoid a PackageGroupConfiguredTarget
    // needing itself for the visibility check. It may work, but I did not
    // think it over completely.
    return ConstantRuleVisibility.PUBLIC;
  }

  @Override
  public boolean isConfigurable() {
    return false;
  }

  public static String targetKind() {
    return "package group";
  }
}
