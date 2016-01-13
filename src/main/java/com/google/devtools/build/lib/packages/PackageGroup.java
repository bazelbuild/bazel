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
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.License.DistributionType;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * This class represents a package group. It has a name and a set of packages
 * and can be asked if a specific package is included in it. The package set is
 * represented as a list of PathFragments.
 */
public class PackageGroup implements Target {
  private boolean containsErrors;
  private final Label label;
  private final Location location;
  private final Package containingPackage;
  private final List<PackageSpecification> packageSpecifications;
  private final List<Label> includes;

  public PackageGroup(Label label, Package pkg, Collection<String> packages,
      Collection<Label> includes, EventHandler eventHandler, Location location) {
    this.label = label;
    this.location = location;
    this.containingPackage = pkg;
    this.includes = ImmutableList.copyOf(includes);

    ImmutableList.Builder<PackageSpecification> packagesBuilder = ImmutableList.builder();
    for (String containedPackage : packages) {
      PackageSpecification specification = null;
      try {
        specification = PackageSpecification.fromString(label, containedPackage);
      } catch (PackageSpecification.InvalidPackageSpecificationException e) {
        containsErrors = true;
        eventHandler.handle(Event.error(location, e.getMessage()));
      }

      if (specification != null) {
        packagesBuilder.add(specification);
      }
    }
    this.packageSpecifications = packagesBuilder.build();
  }

  public boolean containsErrors() {
    return containsErrors;
  }

  public Iterable<PackageSpecification> getPackageSpecifications() {
    return packageSpecifications;
  }

  public boolean contains(Package pkg) {
    for (PackageSpecification specification : packageSpecifications) {
      if (specification.containsPackage(pkg.getPackageIdentifier())) {
        return true;
      }
    }

    return false;
  }

  public List<Label> getIncludes() {
    return includes;
  }

  public List<String> getContainedPackages() {
    List<String> result = Lists.newArrayListWithCapacity(packageSpecifications.size());
    for (PackageSpecification specification : packageSpecifications) {
      result.add(specification.toString());
    }
    return result;
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
