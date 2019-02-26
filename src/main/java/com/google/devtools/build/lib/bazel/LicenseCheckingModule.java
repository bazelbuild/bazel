// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableTable;
import com.google.common.collect.Table;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.LicensesProvider;
import com.google.devtools.build.lib.analysis.LicensesProvider.TargetLicense;
import com.google.devtools.build.lib.analysis.StaticallyLinkedMarkerProvider;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.License;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.packages.License.LicenseType;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.StarlarkSemanticsOptions;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import java.util.EnumSet;
import java.util.Set;

/**
 * Module responsible for checking third party license compliance.
 *
 * <p><b>This is outdated logic marked for removal.</b> See <a
 * href="https://github.com/bazelbuild/bazel/issues/7444">#7444</a> for details.
 */
public class LicenseCheckingModule extends BlazeModule {

  protected boolean shouldCheckLicenses(BuildRequest request, BuildOptions buildOptions) {
    if (request.getOptions(StarlarkSemanticsOptions.class)
        .incompatibleDisableThirdPartyLicenseChecking) {
      return false;
    }
    return buildOptions.get(BuildConfiguration.Options.class).checkLicenses;
  }

  @Override
  public void afterAnalysis(
      CommandEnvironment env,
      BuildRequest request,
      BuildOptions buildOptions,
      Iterable<ConfiguredTarget> configuredTargets)
      throws InterruptedException, ViewCreationFailedException {
    // Check licenses.
    // We check licenses if the first target configuration has license checking enabled. Right
    // now, it is not possible to have multiple target configurations with different settings
    // for this flag, which allows us to take this short cut.
    if (shouldCheckLicenses(request, buildOptions)) {
      Profiler.instance().markPhase(ProfilePhase.LICENSE);
      try (SilentCloseable c = Profiler.instance().profile("validateLicensingForTargets")) {
        validateLicensingForTargets(env, configuredTargets, request.getKeepGoing());
      }
    }
  }

  /**
   * Takes a set of configured targets, and checks if the distribution methods declared for the
   * targets are compatible with the constraints imposed by their prerequisites' licenses.
   *
   * @param configuredTargets the targets to check
   * @param keepGoing if false, and a licensing error is encountered, both generates an error
   *     message on the reporter, <em>and</em> throws an exception. If true, then just generates a
   *     message on the reporter.
   * @throws ViewCreationFailedException if the license checking failed (and not --keep_going)
   */
  private static void validateLicensingForTargets(
      CommandEnvironment env, Iterable<ConfiguredTarget> configuredTargets, boolean keepGoing)
      throws ViewCreationFailedException {
    for (ConfiguredTarget configuredTarget : configuredTargets) {
      Target target = null;
      try {
        target = env.getPackageManager().getTarget(env.getReporter(), configuredTarget.getLabel());
      } catch (NoSuchPackageException | NoSuchTargetException | InterruptedException e) {
        env.getReporter().handle(Event.error("Failed to get target to validate license"));
        throw new ViewCreationFailedException(
            "Build aborted due to issue getting targets to validate licenses", e);
      }

      if (TargetUtils.isTestRule(target)) {
        continue; // Tests are exempt from license checking
      }

      final Set<DistributionType> distribs = target.getDistributions();
      StaticallyLinkedMarkerProvider markerProvider =
          configuredTarget.getProvider(StaticallyLinkedMarkerProvider.class);
      boolean staticallyLinked = markerProvider != null && markerProvider.isLinkedStatically();

      LicensesProvider provider = configuredTarget.getProvider(LicensesProvider.class);
      if (provider != null) {
        NestedSet<TargetLicense> licenses = provider.getTransitiveLicenses();
        for (TargetLicense targetLicense : licenses) {
          if (!checkCompatibility(
              targetLicense.getLicense(),
              distribs,
              target,
              targetLicense.getLabel(),
              env.getReporter(),
              staticallyLinked)) {
            if (!keepGoing) {
              throw new ViewCreationFailedException("Build aborted due to licensing error");
            }
          }
        }
      } else if (target instanceof InputFile) {
        // Input file targets do not provide licenses because they do not
        // depend on the rule where their license is taken from. This is usually
        // not a problem, because the transitive collection of licenses always
        // hits the rule they come from, except when the input file is a
        // top-level target. Thus, we need to handle that case specially here.
        //
        // See FileTarget#getLicense for more information about the handling of
        // license issues with File targets.
        License license = target.getLicense();
        if (!checkCompatibility(
            license,
            distribs,
            target,
            configuredTarget.getLabel(),
            env.getReporter(),
            staticallyLinked)) {
          if (!keepGoing) {
            throw new ViewCreationFailedException("Build aborted due to licensing error");
          }
        }
      }
    }
  }

  private static final Object MARKER = new Object();

  /**
   * The license incompatibility set. This contains the set of (Distribution,License) pairs that
   * should generate errors.
   */
  private static final Table<DistributionType, LicenseType, Object> licenseIncompatibilies =
      createLicenseIncompatibilitySet();

  private static Table<DistributionType, LicenseType, Object> createLicenseIncompatibilitySet() {
    Table<DistributionType, LicenseType, Object> result = HashBasedTable.create();
    result.put(DistributionType.CLIENT, LicenseType.RESTRICTED, MARKER);
    result.put(DistributionType.EMBEDDED, LicenseType.RESTRICTED, MARKER);
    result.put(DistributionType.INTERNAL, LicenseType.BY_EXCEPTION_ONLY, MARKER);
    result.put(DistributionType.CLIENT, LicenseType.BY_EXCEPTION_ONLY, MARKER);
    result.put(DistributionType.WEB, LicenseType.BY_EXCEPTION_ONLY, MARKER);
    result.put(DistributionType.EMBEDDED, LicenseType.BY_EXCEPTION_ONLY, MARKER);
    return ImmutableTable.copyOf(result);
  }

  /**
   * The license warning set. This contains the set of (Distribution,License) pairs that should
   * generate warnings when the user requests verbose license checking.
   */
  private static final Table<DistributionType, LicenseType, Object> licenseWarnings =
      createLicenseWarningsSet();

  private static Table<DistributionType, LicenseType, Object> createLicenseWarningsSet() {
    Table<DistributionType, LicenseType, Object> result = HashBasedTable.create();
    result.put(DistributionType.CLIENT, LicenseType.RECIPROCAL, MARKER);
    result.put(DistributionType.EMBEDDED, LicenseType.RECIPROCAL, MARKER);
    result.put(DistributionType.CLIENT, LicenseType.NOTICE, MARKER);
    result.put(DistributionType.EMBEDDED, LicenseType.NOTICE, MARKER);
    return ImmutableTable.copyOf(result);
  }

  /**
   * Checks if the given license is compatible with distributing a particular target in some set of
   * distribution modes.
   *
   * @param license the license to check
   * @param dists the modes of distribution
   * @param target the target which is being checked, and which will be used for checking exceptions
   * @param licensedTarget the target which declared the license being checked.
   * @param eventHandler a reporter where any licensing issues discovered should be reported
   * @param staticallyLinked whether the target is statically linked under this command
   * @return true if the license is compatible with the distributions
   */
  @VisibleForTesting
  public static boolean checkCompatibility(
      License license,
      Set<DistributionType> dists,
      Target target,
      Label licensedTarget,
      EventHandler eventHandler,
      boolean staticallyLinked) {
    Location location = (target instanceof Rule) ? ((Rule) target).getLocation() : null;

    LicenseType leastRestrictiveLicense;
    if (license.getLicenseTypes().contains(LicenseType.RESTRICTED_IF_STATICALLY_LINKED)) {
      Set<LicenseType> tempLicenses = EnumSet.copyOf(license.getLicenseTypes());
      tempLicenses.remove(LicenseType.RESTRICTED_IF_STATICALLY_LINKED);
      if (staticallyLinked) {
        tempLicenses.add(LicenseType.RESTRICTED);
      } else {
        tempLicenses.add(LicenseType.UNENCUMBERED);
      }
      leastRestrictiveLicense = License.leastRestrictive(tempLicenses);
    } else {
      leastRestrictiveLicense = License.leastRestrictive(license.getLicenseTypes());
    }
    for (DistributionType dt : dists) {
      if (licenseIncompatibilies.contains(dt, leastRestrictiveLicense)) {
        if (!license.getExceptions().contains(target.getLabel())) {
          eventHandler.handle(
              Event.error(
                  location,
                  "Build target '"
                      + target.getLabel()
                      + "' is not compatible with license '"
                      + license
                      + "' from target '"
                      + licensedTarget
                      + "'"));
          return false;
        }
      } else if (licenseWarnings.contains(dt, leastRestrictiveLicense)) {
        eventHandler.handle(
            Event.warn(
                location,
                "Build target '"
                    + target
                    + "' has a potential licensing issue with a '"
                    + license
                    + "' license from target '"
                    + licensedTarget
                    + "'"));
      }
    }
    return true;
  }
}
