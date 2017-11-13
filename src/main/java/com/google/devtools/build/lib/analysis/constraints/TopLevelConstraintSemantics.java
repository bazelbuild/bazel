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

package com.google.devtools.build.lib.analysis.constraints;

import com.google.common.base.Joiner;
import com.google.common.base.Verify;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.configuredtargets.OutputFileConfiguredTarget;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.EnvironmentGroup;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Constraint semantics that apply to top-level targets.
 *
 * <p>Top-level targets are "special" because they have no parents that can assert expected
 * environment compatibility. So these expectations have to be declared by other means.
 *
 * <p>For all other targets see {@link ConstraintSemantics}.
 */
public class TopLevelConstraintSemantics {

  /**
   * Checks that if this is an environment-restricted build, all top-level targets support expected
   * top-level environments. Expected top-level environments can be declared explicitly through
   * {@code --target_environment} or implicitly through {@code --auto_cpu_environment_group}. For
   * the latter, top-level targets must be compatible with the build's target configuration CPU.
   *
   * <p>If any target doesn't support an explicitly expected environment declared through {@link
   * BuildConfiguration.Options#targetEnvironments}, the entire build fails with an error.
   *
   * <p>If any target doesn't support an implicitly expected environment declared through {@link
   * BuildConfiguration.Options#autoCpuEnvironmentGroup}, the target is skipped during execution
   * while remaining targets execute as normal.
   *
   * @param topLevelTargets the build's top-level targets
   * @param packageManager object for retrieving loaded targets
   * @param eventHandler the build's event handler
   * @return the set of bad top-level targets.
   * @throws ViewCreationFailedException if any target doesn't support an explicitly expected
   *     environment declared through {@link BuildConfiguration.Options#targetEnvironments}
   */
  public static Set<ConfiguredTarget> checkTargetEnvironmentRestrictions(
      Iterable<ConfiguredTarget> topLevelTargets,
      PackageManager packageManager,
      ExtendedEventHandler eventHandler)
      throws ViewCreationFailedException, InterruptedException {
    ImmutableSet.Builder<ConfiguredTarget> badTargets = ImmutableSet.builder();
    // Maps targets that are missing *explicitly* required environments to the set of environments
    // they're missing. These targets trigger a ViewCreationFailedException, which halts the build.
    // Targets with missing *implicitly* required environments don't belong here, since the build
    // continues while skipping them.
    Multimap<ConfiguredTarget, Label> exceptionInducingTargets = ArrayListMultimap.create();
    for (ConfiguredTarget topLevelTarget : topLevelTargets) {
      BuildConfiguration config = topLevelTarget.getConfiguration();
      boolean failBuildIfTargetIsBad = true;
      if (config == null) {
        // TODO(bazel-team): support file targets (they should apply package-default constraints).
        continue;
      } else if (!config.enforceConstraints()) {
        continue;
      }

      List<Label> targetEnvironments = config.getTargetEnvironments();
      if (targetEnvironments.isEmpty()) {
        try {
          targetEnvironments = autoConfigureTargetEnvironments(config,
              config.getAutoCpuEnvironmentGroup(), packageManager, eventHandler);
          failBuildIfTargetIsBad = false;
        } catch (NoSuchPackageException
            | NoSuchTargetException
            | ConstraintSemantics.EnvironmentLookupException e) {
          throw new ViewCreationFailedException("invalid target environment", e);
        }
      }

      if (targetEnvironments.isEmpty()) {
        continue;
      }

      // Parse and collect this configuration's environments.
      EnvironmentCollection.Builder builder = new EnvironmentCollection.Builder();
      for (Label envLabel : targetEnvironments) {
        try {
          Target env = packageManager.getTarget(eventHandler, envLabel);
          builder.put(ConstraintSemantics.getEnvironmentGroup(env), envLabel);
        } catch (NoSuchPackageException | NoSuchTargetException
            | ConstraintSemantics.EnvironmentLookupException e) {
          throw new ViewCreationFailedException("invalid target environment", e);
        }
      }
      EnvironmentCollection expectedEnvironments = builder.build();

      // If this target doesn't participate in constraints, ignore it.
      if (!topLevelTarget
          .getTarget()
          .getAssociatedRule()
          .getRuleClassObject()
          .supportsConstraintChecking()) {
        continue;
      }

      // Now check the target against those environments.
      TransitiveInfoCollection asProvider;
      if (topLevelTarget instanceof OutputFileConfiguredTarget) {
        asProvider = ((OutputFileConfiguredTarget) topLevelTarget).getGeneratingRule();
      } else {
        asProvider = topLevelTarget;
      }
      SupportedEnvironmentsProvider provider =
          Verify.verifyNotNull(asProvider.getProvider(SupportedEnvironmentsProvider.class));
      Collection<Label> missingEnvironments =
          ConstraintSemantics.getUnsupportedEnvironments(
              provider.getRefinedEnvironments(), expectedEnvironments);
      if (!missingEnvironments.isEmpty()) {
        badTargets.add(topLevelTarget);
        if (failBuildIfTargetIsBad) {
          exceptionInducingTargets.putAll(topLevelTarget, missingEnvironments);
        }
      }
    }

    if (!exceptionInducingTargets.isEmpty()) {
      throw new ViewCreationFailedException(getBadTargetsUserMessage(exceptionInducingTargets));
    }
    return ImmutableSet.copyOf(badTargets.build());
  }

  /**
   * Helper method for {@link #checkTargetEnvironmentRestrictions} that populates inferred
   * expected environments.
   */
  private static List<Label> autoConfigureTargetEnvironments(BuildConfiguration config,
      @Nullable Label environmentGroupLabel, PackageManager packageManager,
      ExtendedEventHandler eventHandler)
      throws InterruptedException, NoSuchTargetException, NoSuchPackageException,
      ConstraintSemantics.EnvironmentLookupException {
    if (environmentGroupLabel == null) {
      return ImmutableList.of();
    }

    EnvironmentGroup environmentGroup = (EnvironmentGroup)
        packageManager.getTarget(eventHandler, environmentGroupLabel);

    ImmutableList.Builder<Label> targetEnvironments = new ImmutableList.Builder<>();
    for (Label environmentLabel : environmentGroup.getEnvironments()) {
      if (environmentLabel.getName().equals(config.getCpu())) {
        targetEnvironments.add(environmentLabel);
      }
    }

    return targetEnvironments.build();
  }

  /**
   * Prepares a user-friendly error message for a list of targets missing support for required
   * environments.
   */
  private static String getBadTargetsUserMessage(Multimap<ConfiguredTarget, Label> badTargets) {
    StringBuilder msg = new StringBuilder();
    msg.append("This is a restricted-environment build.");
    for (Map.Entry<ConfiguredTarget, Collection<Label>> entry : badTargets.asMap().entrySet()) {
      msg.append(String.format("\n - %s does not support required environment%s %s.",
          entry.getKey().getLabel(),
          entry.getValue().size() == 1 ? "" : "s",
          Joiner.on(", ").join(entry.getValue())));
    }
    return msg.toString();
  }
}
