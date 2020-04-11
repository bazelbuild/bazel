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

import com.google.common.base.Predicates;
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
import com.google.devtools.build.lib.analysis.constraints.SupportedEnvironmentsProvider.RemovedEnvironmentCulprit;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.EnvironmentGroup;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue.Key;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;
import java.util.function.Function;
import java.util.stream.Collectors;
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
  private final PackageManager packageManager;
  private final Function<Key, BuildConfiguration> configurationProvider;
  private final ExtendedEventHandler eventHandler;

  /**
   * Constructor with helper classes for loading targets.
   *
   * @param packageManager object for retrieving loaded targets
   * @param eventHandler the build's event handler
   */
  public TopLevelConstraintSemantics(
      PackageManager packageManager,
      Function<Key, BuildConfiguration> configurationProvider,
      ExtendedEventHandler eventHandler) {
    this.packageManager = packageManager;
    this.configurationProvider = configurationProvider;
    this.eventHandler = eventHandler;
  }

  private static class MissingEnvironment {
    private final Label environment;
    @Nullable
    // If null, the top-level target just didn't declare a required environment. If not null, that
    // means the declaration got "refined" away due to some select() somewhere in its deps. See
    // ConstraintSemantics's documentation for an explanation of refinement.
    private final RemovedEnvironmentCulprit culprit;
    private MissingEnvironment(Label environment, RemovedEnvironmentCulprit culprit) {
      this.environment = environment;
      this.culprit = culprit;
    }
  }

  /**
   * Checks that if this is an environment-restricted build, all top-level targets support expected
   * top-level environments. Expected top-level environments can be declared explicitly through
   * {@code --target_environment} or implicitly through {@code --auto_cpu_environment_group}. For
   * the latter, top-level targets must be compatible with the build's target configuration CPU.
   *
   * <p>If any target doesn't support an explicitly expected environment declared through {@link
   * CoreOptions#targetEnvironments}, the entire build fails with an error.
   *
   * <p>If any target doesn't support an implicitly expected environment declared through {@link
   * CoreOptions#autoCpuEnvironmentGroup}, the target is skipped during execution while remaining
   * targets execute as normal.
   *
   * @param topLevelTargets the build's top-level targets
   * @return the set of bad top-level targets.
   * @throws ViewCreationFailedException if any target doesn't support an explicitly expected
   *     environment declared through {@link CoreOptions#targetEnvironments}
   */
  public Set<ConfiguredTarget> checkTargetEnvironmentRestrictions(
      ImmutableList<ConfiguredTarget> topLevelTargets)
      throws ViewCreationFailedException, InterruptedException {
    ImmutableSet.Builder<ConfiguredTarget> badTargets = ImmutableSet.builder();
    // Maps targets that are missing *explicitly* required environments to the set of environments
    // they're missing. These targets trigger a ViewCreationFailedException, which halts the build.
    // Targets with missing *implicitly* required environments don't belong here, since the build
    // continues while skipping them.
    Multimap<ConfiguredTarget, MissingEnvironment> exceptionInducingTargets =
        ArrayListMultimap.create();
    for (ConfiguredTarget topLevelTarget : topLevelTargets) {
      BuildConfiguration config = configurationProvider.apply(topLevelTarget.getConfigurationKey());
      Target target = null;
      try {
        target = packageManager.getTarget(eventHandler, topLevelTarget.getLabel());
      } catch (NoSuchPackageException | NoSuchTargetException e) {
        eventHandler.handle(
            Event.error(
                "Unable to get target from package when checking environment restrictions. " + e));
        continue;
      }
      if (config == null) {
        // TODO(bazel-team): support file targets (they should apply package-default constraints).
        continue;
      } else if (!config.enforceConstraints()) {
        continue;  // Constraint checking is disabled for all targets.
      } else if (target.getAssociatedRule() == null) {
        continue;
      } else if (!target.getAssociatedRule().getRuleClassObject().supportsConstraintChecking()) {
        continue; // This target doesn't participate in constraints.
      }

      // Check explicitly expected environments.
      exceptionInducingTargets.putAll(topLevelTarget, // This is a no-op on empty collections.
          getMissingEnvironments(topLevelTarget, config.getTargetEnvironments()));

      // Check auto-detected CPU environments.
      try {
        if (!getMissingEnvironments(topLevelTarget,
            autoConfigureTargetEnvironments(config, config.getAutoCpuEnvironmentGroup()))
            .isEmpty()) {
          badTargets.add(topLevelTarget);
        }
      } catch (NoSuchPackageException
          | NoSuchTargetException e) {
        throw new ViewCreationFailedException("invalid target environment", e);
      }
    }

    if (!exceptionInducingTargets.isEmpty()) {
      throw new ViewCreationFailedException(getBadTargetsUserMessage(exceptionInducingTargets));
    }
    return ImmutableSet.copyOf(
        badTargets
            .addAll(exceptionInducingTargets.keySet())
            .build());
  }

  /**
   * Helper method for {@link #checkTargetEnvironmentRestrictions} that populates inferred
   * expected environments.
   */
  private List<Label> autoConfigureTargetEnvironments(BuildConfiguration config,
      @Nullable Label environmentGroupLabel)
      throws InterruptedException, NoSuchTargetException, NoSuchPackageException {
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
   * Returns the expected environments that the given top-level target doesn't support.
   *
   * @param topLevelTarget the top-level target to check
   * @param expectedEnvironmentLabels the environments this target is expected to support
   *
   * @throw InterruptedException if environment target resolution fails
   * @throw ViewCreationFailedException if an expected environment isn't a valid target
   */
  private Collection<MissingEnvironment> getMissingEnvironments(ConfiguredTarget topLevelTarget,
      Collection<Label> expectedEnvironmentLabels)
      throws InterruptedException, ViewCreationFailedException {
    if (expectedEnvironmentLabels.isEmpty()) {
      return ImmutableList.of();
    }

    // Convert expected environment labels to actual environments.
    EnvironmentCollection.Builder expectedEnvironmentsBuilder = new EnvironmentCollection.Builder();
    for (Label envLabel : expectedEnvironmentLabels) {
      try {
        Target env = packageManager.getTarget(eventHandler, envLabel);
        expectedEnvironmentsBuilder.put(
            ConstraintSemantics.getEnvironmentGroup(env).getEnvironmentLabels(), envLabel);
      } catch (NoSuchPackageException | NoSuchTargetException
          | ConstraintSemantics.EnvironmentLookupException e) {
        throw new ViewCreationFailedException("invalid target environment", e);
      }
    }
    EnvironmentCollection expectedEnvironments = expectedEnvironmentsBuilder.build();

    // Dereference any aliases that might be present.
    topLevelTarget = topLevelTarget.getActual();
    // Now check the target against expected environments.
    TransitiveInfoCollection asProvider;
    if (topLevelTarget instanceof OutputFileConfiguredTarget) {
      asProvider = ((OutputFileConfiguredTarget) topLevelTarget).getGeneratingRule();
    } else {
      asProvider = topLevelTarget;
    }
    SupportedEnvironmentsProvider provider =
        Verify.verifyNotNull(asProvider.getProvider(SupportedEnvironmentsProvider.class));
    return ConstraintSemantics
        .getUnsupportedEnvironments(provider.getRefinedEnvironments(), expectedEnvironments)
        .stream()
        // We apply this filter because the target might also not support default environments in
        // other environment groups. We don't care about those. We only care about the environments
        // explicitly referenced.
        .filter(Predicates.in(expectedEnvironmentLabels))
        .map(environment ->
            new MissingEnvironment(environment, provider.getRemovedEnvironmentCulprit(environment)))
        .collect(Collectors.toSet());
  }

  /**
   * Prepares a user-friendly error message for a list of targets missing support for required
   * environments.
   */
  private static String getBadTargetsUserMessage(Multimap<ConfiguredTarget,
      MissingEnvironment> badTargets) {
    StringJoiner msg = new StringJoiner("\n");
    msg.add("This is a restricted-environment build.");
    for (Map.Entry<ConfiguredTarget, Collection<MissingEnvironment>> entry :
        badTargets.asMap().entrySet()) {
      msg
          .add(" ")
          .add(entry.getKey().getLabel() + " does not support:");
      boolean isFirst = true;
      boolean lastEntryWasMultiline = false;
      for (MissingEnvironment missingEnvironment : entry.getValue()) {
        if (missingEnvironment.culprit == null) {
          // The target didn't declare support for this environment.
          if (lastEntryWasMultiline) {
            // Pretty-format: if the last environment message was multi-line, make it clear this
            // one is a different entry. But we don't want to do that if all entries are single-line
            // because that would be pointlessly long.
            msg.add(" ");
          }
          msg.add("  " + missingEnvironment.environment);
          lastEntryWasMultiline = false;
        } else {
          // The target declared support, but it was refined out by a select() somewhere in its
          // transitive deps.
          if (!isFirst) {
            msg.add(" "); // Pretty-format for clarity.
          }
          msg.add(
              ConstraintSemantics.getMissingEnvironmentCulpritMessage(
                  missingEnvironment.environment, missingEnvironment.culprit));
          lastEntryWasMultiline = true;
        }
        isFirst = false;
      }
    }
    return msg.add(" ").toString();
  }
}
