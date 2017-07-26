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

package com.google.devtools.build.lib.analysis.featurecontrol;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.devtools.build.lib.analysis.RedirectChaser;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.PackageSpecification;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.ArrayDeque;
import java.util.LinkedHashSet;
import java.util.Queue;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * A loader for the FeaturePolicyConfiguration fragment.
 *
 * @deprecated This is deprecated because the dependency on the package group used to hold the
 *     whitelist is not accessible through blaze query. Use {@link Whitelist}.
 */
@Deprecated
public final class FeaturePolicyLoader implements ConfigurationFragmentFactory {

  private final ImmutableSet<String> permittedFeatures;

  public FeaturePolicyLoader(Iterable<String> permittedFeatures) {
    this.permittedFeatures = ImmutableSet.copyOf(permittedFeatures);
    for (String permittedFeature : this.permittedFeatures) {
      Preconditions.checkArgument(
          !permittedFeature.contains("="), "Feature names cannot contain =");
    }
  }

  @Override
  @Nullable
  public BuildConfiguration.Fragment create(ConfigurationEnvironment env, BuildOptions buildOptions)
      throws InvalidConfigurationException, InterruptedException {
    ImmutableSetMultimap.Builder<String, PackageSpecification> features =
        new ImmutableSetMultimap.Builder<>();
    ImmutableMap.Builder<String, String> policies = new ImmutableMap.Builder<>();
    LinkedHashSet<String> unseenFeatures = new LinkedHashSet<>(permittedFeatures);

    for (PolicyEntry entry : buildOptions.get(FeaturePolicyOptions.class).policies) {
      if (!permittedFeatures.contains(entry.getFeature())) {
        // It would be so nice to be able to do this during parsing... but because options are
        // parsed statically through reflection, we have no way of doing this until we get to the
        // configuration loader.
        throw new InvalidConfigurationException("No such feature: " + entry.getFeature());
      }
      if (!unseenFeatures.remove(entry.getFeature())) {
        // TODO(mstaib): Perhaps we should allow for overriding or concatenation here?
        throw new InvalidConfigurationException(
            "Multiple definitions of the rollout policy for feature "
                + entry.getFeature()
                + ". To use multiple package_groups, use a package_group with the includes "
                + "attribute instead.");
      }

      Iterable<PackageSpecification> packageSpecifications =
          getAllPackageSpecificationsForPackageGroup(
              env, entry.getPackageGroupLabel(), entry.getFeature());

      if (packageSpecifications == null) {
        return null;
      }

      features.putAll(entry.getFeature(), packageSpecifications);
      policies.put(entry.getFeature(), entry.getPackageGroupLabel().toString());
    }

    // Default to universal access for all features not declared.
    for (String unseenFeature : unseenFeatures) {
      features.put(unseenFeature, PackageSpecification.everything());
      policies.put(unseenFeature, "//...");
    }

    return new FeaturePolicyConfiguration(features.build(), policies.build());
  }

  /**
   * Evaluates, recursively, the given package group. Returns {@code null} in the case of missing
   * Skyframe dependencies.
   */
  @Nullable
  private static Iterable<PackageSpecification> getAllPackageSpecificationsForPackageGroup(
      ConfigurationEnvironment env, Label packageGroupLabel, String feature)
      throws InvalidConfigurationException, InterruptedException {
    String context = feature + " feature policy";
    Label actualPackageGroupLabel = RedirectChaser.followRedirects(env, packageGroupLabel, context);
    if (actualPackageGroupLabel == null) {
      return null;
    }

    ImmutableSet.Builder<PackageSpecification> packages = new ImmutableSet.Builder<>();
    Set<Label> alreadyVisitedPackageGroups = new LinkedHashSet<>();
    Queue<Label> packageGroupsToVisit = new ArrayDeque<>();
    packageGroupsToVisit.add(actualPackageGroupLabel);
    alreadyVisitedPackageGroups.add(actualPackageGroupLabel);

    while (!packageGroupsToVisit.isEmpty()) {
      Target target;
      try {
        // This is guaranteed to succeed, because the RedirectChaser was used to get this label,
        // and it will throw an InvalidConfigurationException if the target doesn't exist.
        target = env.getTarget(packageGroupsToVisit.remove());
      } catch (NoSuchPackageException | NoSuchTargetException impossible) {
        throw new AssertionError(impossible);
      }
      if (target == null) {
        return null;
      }

      if (!(target instanceof PackageGroup)) {
        throw new InvalidConfigurationException(
            target.getLabel() + " is not a package_group in " + context);
      }

      PackageGroup packageGroup = (PackageGroup) target;

      packages.addAll(packageGroup.getPackageSpecifications());

      for (Label include : packageGroup.getIncludes()) {
        Label actualInclude = RedirectChaser.followRedirects(env, include, context);
        if (actualInclude == null) {
          return null;
        }

        if (alreadyVisitedPackageGroups.add(actualInclude)) {
          packageGroupsToVisit.add(actualInclude);
        }
      }
    }

    return packages.build();
  }

  @Override
  public Class<? extends BuildConfiguration.Fragment> creates() {
    return FeaturePolicyConfiguration.class;
  }

  @Override
  public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
    return ImmutableSet.<Class<? extends FragmentOptions>>of(FeaturePolicyOptions.class);
  }
}
