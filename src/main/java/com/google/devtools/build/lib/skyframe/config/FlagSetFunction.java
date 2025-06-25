// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.config;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.devtools.common.options.OptionsParser.STARLARK_SKIPPED_PREFIXES;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Splitter;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.runtime.ConfigFlagDefinitions;
import com.google.devtools.build.lib.skyframe.ProjectValue;
import com.google.devtools.build.lib.skyframe.ProjectValue.BuildableUnit;
import com.google.devtools.build.lib.skyframe.ProjectValue.EnforcementPolicy;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.common.options.GlobalRcUtils;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * A SkyFunction that, given an scl file path and the name of scl configs, does the following:
 *
 * <ol>
 *   <li>calls {@link com.google.devtools.build.lib.skyframe.ProjectFunction} to load the content of
 *       scl files given the provided scl config name
 *   <li>calls {@link ParsedFlagsFunction} to parse the list of options
 *   <li>returns the list of flags in command line format to be applied to the build
 * </ol>
 *
 * <p>If --enforce_project_configs is set, invalid --scl_config values or invalid project files will
 * cause the build to fail.
 */
public final class FlagSetFunction implements SkyFunction {
  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws FlagSetFunctionException, InterruptedException {
    FlagSetValue.Key key = (FlagSetValue.Key) skyKey.argument();
    if (!key.enforceCanonical()) {
      if (!key.sclConfig().isEmpty()) {
        env.getListener()
            .handle(
                Event.info(
                    String.format(
                        "Ignoring --scl_config=%s because --enforce_project_configs is not set",
                        key.sclConfig())));
      }
      // --noenforce_project_configs. Nothing to do.
      return FlagSetValue.create(ImmutableSet.of(), ImmutableSet.of());
    }
    ProjectValue projectValue =
        (ProjectValue) env.getValue(new ProjectValue.Key(key.projectFile()));
    if (projectValue == null) {
      return null;
    }

    // Skyframe doesn't replay warnings or info messages on cache hits: see Event.storeForReplay and
    // Reportable.storeForReplay. We want some flag set messages to be more persistent, so we
    // return them in the Skyvalue for the caller to emit.
    ImmutableSet.Builder<Event> persistentMessages = ImmutableSet.builder();
    ImmutableSet<String> sclConfigAsStarlarkList =
        getSclConfig(key, projectValue, persistentMessages, key.targets());
    return FlagSetValue.create(sclConfigAsStarlarkList, persistentMessages.build());
  }

  /**
   * Given an .scl file and {@code --scl_config} value, returns the flags denoted by that {@code
   * --scl_config}. Flags are a list of strings (not parsed through the options parser).
   */
  private static ImmutableSet<String> getSclConfig(
      FlagSetValue.Key key,
      ProjectValue sclContent,
      ImmutableSet.Builder<Event> persistentMessages,
      Set<Label> targets)
      throws FlagSetFunctionException {
    Label projectFile = key.projectFile();
    String sclConfigName = key.sclConfig();
    EnforcementPolicy enforcementPolicy = sclContent.getEnforcementPolicy();

    ImmutableMap<String, ProjectValue.BuildableUnit> configs = sclContent.getBuildableUnits();
    if (configs == null || configs.isEmpty()) {
      // This project file doesn't define configs, so it must not be used for canonical configs.
      return ImmutableSet.of();
    }

    String sclConfigNameForMessage = sclConfigName;
    ImmutableList<String> sclConfigValue = null;
    if (sclConfigName.isEmpty()) {
      // If there's no --scl_config, try to use the default_config.
      ImmutableMap<String, ProjectValue.BuildableUnit> buildableUnits =
          sclContent.getBuildableUnits();

      ImmutableList<ProjectValue.BuildableUnit> defaultBuildableUnits =
          filterProjects(targets, buildableUnits);

      // check that all targets resolves to the same set of flags.
      // orders of flags should not matter here.
      ImmutableSet<ProjectValue.BuildableUnit> resolvedDefaultBuildableUnit =
          resolveSingleMatchingDefaultBuildableUnitForAllTargets(defaultBuildableUnits);
      if (resolvedDefaultBuildableUnit.size() > 1) {
        throw new FlagSetFunctionException(
            new UnsupportedConfigException(
                "Building target(s) with different configurations are not supported."),
            Transience.PERSISTENT);
      }

      if (resolvedDefaultBuildableUnit.isEmpty()) {
        throw new FlagSetFunctionException(
            new UnsupportedConfigException(
                String.format(
                    "This project's builds must set --scl_config because no default config is"
                        + " defined.\n"
                        + "%s",
                    supportedConfigsDesc(projectFile, configs))),
            Transience.PERSISTENT);
      }
      ProjectValue.BuildableUnit buildableUnit = resolvedDefaultBuildableUnit.iterator().next();
      sclConfigValue = buildableUnit.flags();
      sclConfigNameForMessage = buildableUnit.description();
    } else {
      if (!configs.containsKey(sclConfigName)) {
        // The user set --scl_config to an unknown config.
        throw new FlagSetFunctionException(
            new UnsupportedConfigException(
                String.format(
                    "--scl_config=%s is not a valid configuration for this project.%s",
                    sclConfigName, supportedConfigsDesc(projectFile, configs))),
            Transience.PERSISTENT);
      }
      sclConfigValue = configs.get(sclConfigName).flags();
    }

    // Replace --config=foo entries with their expanded definitions.
    sclConfigValue = expandConfigFlags(sclConfigName, sclConfigValue, key.configFlagDefinitions());

    ImmutableList<String> buildOptionsAsStrings = getBuildOptionsAsStrings(key.targetOptions());
    ImmutableSet<String> optionsToApply = filterOptions(sclConfigValue, buildOptionsAsStrings);

    if (optionsToApply.isEmpty()) {
      return ImmutableSet.of();
    }

    Collection<String> alwaysAllowedConfigs =
        sclContent.getAlwaysAllowedConfigs() == null
            ? ImmutableList.of()
            : sclContent.getAlwaysAllowedConfigs();

    validateNoExtraFlagsSet(
        enforcementPolicy,
        alwaysAllowedConfigs,
        buildOptionsAsStrings,
        key.userOptions(),
        optionsToApply,
        persistentMessages,
        projectFile);
    persistentMessages.add(
        Event.info(
            String.format(
                "Applying flags from the config '%s' defined in %s: %s ",
                sclConfigNameForMessage, projectFile, optionsToApply)));
    return optionsToApply;
  }

  /**
   * Returns all default {@link BuildableUnit buildable units} that contain the specific target
   * in the {@code targetPatterns} field. If there are multiple matching default buildable units, an
   * exception will be thrown.
   */
  private static ImmutableList<ProjectValue.BuildableUnit> filterProjects(
      Set<Label> targets, ImmutableMap<String, ProjectValue.BuildableUnit> buildableUnits)
      throws FlagSetFunctionException {
    Map<Label, ProjectValue.BuildableUnit> targetsAndMatchingDefaultBuildableUnits =
        new HashMap<>();
    for (Label target : targets) {
      for (ProjectValue.BuildableUnit buildableUnit : buildableUnits.values()) {
        if (doesBuildableUnitMatchTarget(buildableUnit, target)) {
          if (buildableUnit.isDefault()) {
            if (targetsAndMatchingDefaultBuildableUnits.put(target, buildableUnit) != null) {
              throw new FlagSetFunctionException(
                  new UnsupportedConfigException(
                      String.format(
                          "Multiple matching default configs found for target %s. Please check your"
                              + " project file and ensure that for target %s, there should be only"
                              + " 1 matching default config.",
                          target, target)),
                  Transience.PERSISTENT);
            }
          }
        }
      }
    }

    return ImmutableList.copyOf(targetsAndMatchingDefaultBuildableUnits.values());
  }

  /**
   * Takes a list of default buildable units and compares the flags values of all buildable units.
   * If the flags from all buildable units are the same, returns the first matching buildable unit.
   * Else returns the first matching buildable unit for each distinct set of flags.
   *
   * <p>The caller should check that there are no more than 1 buildable unit returned.
   */
  private static ImmutableSet<ProjectValue.BuildableUnit>
      resolveSingleMatchingDefaultBuildableUnitForAllTargets(
          ImmutableList<ProjectValue.BuildableUnit> defaultBuildableUnitsForAllTargets) {
    LinkedHashMap<ImmutableList<String>, ProjectValue.BuildableUnit> flagsToFirstBuildableUnit =
        new LinkedHashMap<>();
    for (ProjectValue.BuildableUnit buildableUnit : defaultBuildableUnitsForAllTargets) {
      flagsToFirstBuildableUnit.putIfAbsent(buildableUnit.flags(), buildableUnit);
    }
    return ImmutableSet.copyOf(flagsToFirstBuildableUnit.values());
  }

  /**
   * Returns {@code true} iff the {@code specificTarget} matches the target patterns in the {@link
   * BuildableUnit}.
   */
  @VisibleForTesting
  static boolean doesBuildableUnitMatchTarget(BuildableUnit buildableUnit, Label specificTarget) {
    if (buildableUnit.targetPatternMatcher().isEmpty()) {
      return true;
    }
    return buildableUnit.targetPatternMatcher().contains(specificTarget);
  }

  private static ImmutableList<String> getBuildOptionsAsStrings(BuildOptions targetOptions) {
    ImmutableList.Builder<String> allOptionsAsStringsBuilder = new ImmutableList.Builder<>();

    // Collect a list of BuildOptions, excluding TestOptions.
    targetOptions.getStarlarkOptions().keySet().stream()
        .map(Object::toString)
        .forEach(allOptionsAsStringsBuilder::add);
    for (FragmentOptions fragmentOptions : targetOptions.getNativeOptions()) {
      if (fragmentOptions.getClass().equals(TestConfiguration.TestOptions.class)) {
        continue;
      }
      fragmentOptions.asMap().keySet().forEach(allOptionsAsStringsBuilder::add);
    }
    return allOptionsAsStringsBuilder.build();
  }

  /**
   * Filters the options from the selected config to only those that are part of {@link
   * BuildOptions}, excluding {@link TestConfiguration.TestOptions}.
   *
   * <p>Only the options that are part of {@link BuildOptions} are allowed to be set in the project
   * file.
   */
  private static ImmutableSet<String> filterOptions(
      Collection<String> flagsFromSelectedConfig, ImmutableList<String> buildOptionsAsStrings) {
    ImmutableSet.Builder<String> filteredFlags = ImmutableSet.builder();
    for (String flagSetting : flagsFromSelectedConfig) {
      // Remove options that aren't part of BuildOptions from the selected config.
      if (buildOptionsAsStrings.contains(
          Iterables.get(Splitter.on("=").split(flagSetting), 0)
              .replaceFirst("--", "")
              .replace("'", ""))) {
        filteredFlags.add(flagSetting);
      } else if (STARLARK_SKIPPED_PREFIXES.stream().anyMatch(flagSetting::startsWith)) {
        // Because the BuildOptions might not already include Starlark flags that are set in the
        // flagset, explicitly add them to the set of options to return.
        filteredFlags.add(flagSetting);
      }
    }
    return filteredFlags.build();
  }

  /**
   * In-place expands {@code --config=foo} entries in {@code inputFlags}.
   *
   * <p>Doesn't parse flags or check where they're defined. It's up to callers to determine if flags
   * are, for example, part of {@link BuildOptions}, if they parse correctly, or if they even exist.
   *
   * @throws FlagSetFunctionException if {@code --config=foo} doesn't evaluate, it defines
   *     non-{@link BuildOptions} flags, isn't defined in a global rc file, or is defined multiple
   *     times.
   */
  private static ImmutableList<String> expandConfigFlags(
      String sclConfigName,
      Collection<String> inputFlags,
      ConfigFlagDefinitions configFlagDefinitions)
      throws FlagSetFunctionException {
    // First look for dupes.
    HashSet<String> dupeChecker = new HashSet<>();
    for (var flag : inputFlags) {
      if (flag.startsWith("--config=") && !dupeChecker.add(flag)) {
        throw new FlagSetFunctionException(
            new UnsupportedConfigException(
                String.format(
                    "--scl_config=%s: %s appears multiple times. Please ensure it appears at most"
                        + " once.",
                    sclConfigName, flag)),
            Transience.PERSISTENT);
      }
    }

    // Now rebuild the input list while in-place expanding each "--config=foo" entry.
    var ans = ImmutableList.<String>builder();
    for (var flag : inputFlags) {
      if (!flag.startsWith("--config=")) {
        ans.add(flag);
        continue;
      }
      // TODO: b/388289978 - fail when a --config sets non-BuildOptions flags.
      ConfigFlagDefinitions.ConfigValue expandedFlags;
      try {
        expandedFlags =
            ConfigFlagDefinitions.get(flag.substring(flag.indexOf("=") + 1), configFlagDefinitions);
      } catch (OptionsParsingException e) {
        throw new FlagSetFunctionException(
            new UnsupportedConfigException(
                String.format("--scl_config=%s: %s", sclConfigName, e.getMessage())),
            Transience.PERSISTENT);
      }
      for (String rcSource : expandedFlags.rcSources()) {
        if (!GlobalRcUtils.isGlobalRcFile(rcSource)) {
          throw new FlagSetFunctionException(
              new UnsupportedConfigException(
                  String.format(
                      "--scl_config=%s: can't set %s because its definition depends on %s which"
                          + " isn't a global rc file.",
                      sclConfigName, flag, rcSource)),
              Transience.PERSISTENT);
        }
      }
      ans.addAll(expandedFlags.flags());
    }
    return ans.build();
  }

  /**
   * Enforces one of the following `enforcement_policies`:
   *
   * <p>WARN - warn if the user set any output-affecting options that are not present in the
   * selected config in a bazelrc or on the command line.
   *
   * <p>COMPATIBLE - fail if the user set any options that are present in the selected config to a
   * different value than the one in the config. Also warn for other output-affecting options
   *
   * <p>STRICT - fail if the user set any output-affecting options that are not present in the
   * selected config.
   *
   * <p>Conflicting output-affecting options may be set in global RC files (including the {@code
   * InvocationPolicy}). Flags that do not affect outputs are always allowed.
   *
   * @param userOptions the user options set in the command line or user bazelrc as a map from
   *     option.getCanonicalForm()to option.getExpandedFrom(), {"--define=foo=bar": "--config=foo"}.
   */
  private static void validateNoExtraFlagsSet(
      EnforcementPolicy enforcementPolicy,
      Collection<String> alwaysAllowedConfigs,
      ImmutableList<String> buildOptionsAsStrings,
      ImmutableMap<String, String> userOptions,
      ImmutableSet<String> flagsFromSelectedConfig,
      ImmutableSet.Builder<Event> persistentMessages,
      Label projectFile)
      throws FlagSetFunctionException {
    ImmutableSet<String> overlap =
        userOptions.keySet().stream()
            // Remove options that aren't part of BuildOptions. This section can be removed once
            // we only include BuildOptions in the passed userOptions.
            .filter(
                option ->
                    buildOptionsAsStrings.contains(
                        Iterables.get(Splitter.on("=").split(option), 0)
                            .replaceFirst("--", "")
                            .replace("'", "")))
            .filter(option -> !option.startsWith("--scl_config"))
            .filter(option -> !flagsFromSelectedConfig.contains(option))
            // Remove options that are expanded from always-allowed configs either defined in the
            // project file...
            .filter(option -> !alwaysAllowedConfigs.contains(userOptions.get(option)))
            // ... or globally
            .filter(
                option -> !GlobalRcUtils.ALLOWED_GLOBAL_CONFIGS.contains(userOptions.get(option)))
            .map(
                option ->
                    userOptions.get(option).isEmpty()
                        ? "'" + option + "'"
                        : "'" + userOptions.get(option) + "'")
            .collect(toImmutableSet());
    if (overlap.isEmpty()) {
      return;
    }
    switch (enforcementPolicy) {
      case WARN:
        break;
      case COMPATIBLE:
        ImmutableSet<String> optionNamesFromSelectedConfig =
            flagsFromSelectedConfig.stream()
                .map(flag -> Iterables.get(Splitter.on("=").split(flag), 0).replace("'", ""))
                .collect(toImmutableSet());
        ImmutableSet<String> conflictingOptions =
            overlap.stream()
                .filter(
                    option ->
                        optionNamesFromSelectedConfig.contains(
                            Iterables.get(Splitter.on("=").split(option), 0).replace("'", "")))
                .collect(toImmutableSet());
        if (!conflictingOptions.isEmpty()) {
          throw new FlagSetFunctionException(
              new UnsupportedConfigException(
                  String.format(
                      "This build uses a project file (%s) that does not allow conflicting flags"
                          + " in the command line or user bazelrc. Found %s. Please remove these"
                          + " flags or disable project file resolution via"
                          + " --noenforce_project_configs.",
                      projectFile, conflictingOptions)),
              Transience.PERSISTENT);
        }
        break;
      case STRICT:
        throw new FlagSetFunctionException(
            new UnsupportedConfigException(
                String.format(
                    "This build uses a project file (%s) that does not allow output-affecting"
                        + " flags in the command line or user bazelrc. Found %s. Please remove"
                        + " these flags or disable project file resolution via"
                        + " --noenforce_project_configs.",
                    projectFile, overlap)),
            Transience.PERSISTENT);
    }
    // This appears in the WARN case, or for a COMPATIBLE project file that doesn't have
    // conflicting flags. We never hit this in the STRICT case, since we've already thrown.
    persistentMessages.add(
        Event.warn(
            String.format(
                "This build uses a project file (%s), but also sets output-affecting"
                    + " flags in the command line or user bazelrc: %s. Please consider"
                    + " removing these flags.",
                projectFile, overlap)));
  }

  /** Returns a user-friendly description of project-supported configurations. */
  private static String supportedConfigsDesc(
      Label projectFile, Map<String, ProjectValue.BuildableUnit> configs) {
    String ans = "\nThis project supports:\n";
    int longestNameLength =
        configs.keySet().stream().map(String::length).max(Integer::compareTo).get();
    for (var configInfo : configs.entrySet()) {
      ans +=
          String.format(
              "  --scl_config=%s -> ", Strings.padEnd(configInfo.getKey(), longestNameLength, ' '));
      String desc = configInfo.getValue().description();
      // Add user-friendly description if specified, else list of applied flags.
      ans +=
          desc.isEmpty() || desc.equals(configInfo.getKey())
              ? String.format("[%s]", String.join(" ", configInfo.getValue().flags()))
              : desc;
      ans += "\n";
    }
    ans += String.format("\nThis policy is defined in %s.\n", projectFile.toPathFragment());
    return ans;
  }

  private static final class FlagSetFunctionException extends SkyFunctionException {
    FlagSetFunctionException(Exception cause, Transience transience) {
      super(cause, transience);
    }
  }

  private static final class UnsupportedConfigException extends Exception {
    UnsupportedConfigException(String msg) {
      super(msg);
    }
  }
}
