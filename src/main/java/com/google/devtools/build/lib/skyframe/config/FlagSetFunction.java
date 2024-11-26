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

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.Label.RepoContext;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.skyframe.ProjectValue;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;

/**
 * A SkyFunction that, given an scl file path and the name of scl configs, does the following:
 *
 * <ol>
 *   <li>calls {@link com.google.devtools.build.lib.skyframe.ProjectFunction} to load the content of
 *       scl files given the provided scl config name
 *   <li>calls {@link ParsedFlagsFunction} to parse the list of options
 *   <li>defines a patch transition and applies the transition to the input {@link BuildOptions}
 * </ol>
 *
 * <p>If given an unknown {@link CoreOptions#sclConfig}, {@link FlagSetFunction} returns the
 * original {@link BuildOptions} and doesn't error out.
 */
public final class FlagSetFunction implements SkyFunction {
  private static final String CONFIGS = "configs";

  private static final String DEFAULT_CONFIG = "default_config";

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws FlagSetFunctionException, InterruptedException {
    FlagSetValue.Key key = (FlagSetValue.Key) skyKey.argument();
    if (!key.enforceCanonical()) {
      if (!key.getSclConfig().isEmpty()) {
        env.getListener()
            .handle(
                Event.info(
                    String.format(
                        "Ignoring --scl_config=%s because --enforce_project_configs is not set",
                        key.getSclConfig())));
      }
      // --noenforce_project_configs. Nothing to do.
      return FlagSetValue.create(key.getTargetOptions());
    }
    ProjectValue projectValue =
        (ProjectValue) env.getValue(new ProjectValue.Key(key.getProjectFile()));
    if (projectValue == null) {
      return null;
    }

    ImmutableList<String> sclConfigAsStarlarkList =
        getSclConfig(
            key.getProjectFile(),
            projectValue,
            key.getSclConfig(),
            env.getListener(),
            key.getTargetOptions(),
            key.getUserOptions());
    ParsedFlagsValue parsedFlags = parseFlags(sclConfigAsStarlarkList, env);
    if (parsedFlags == null) {
      return null;
    }
    BuildOptions mergedOptions = parsedFlags.mergeWith(key.getTargetOptions()).getOptions();
    return FlagSetValue.create(mergedOptions);
  }

  /**
   * Given an .scl file and {@code --scl_config} value, returns the flags denoted by that {@code
   * --scl_config}. Flags are a list of strings (not parsed through the options parser).
   */
  @SuppressWarnings("unchecked")
  private ImmutableList<String> getSclConfig(
      Label projectFile,
      ProjectValue sclContent,
      String sclConfigName,
      ExtendedEventHandler eventHandler,
      BuildOptions targetOptions,
      ImmutableMap<String, String> userOptions)
      throws FlagSetFunctionException {

    var configs = (Dict<String, Collection<String>>) sclContent.getResidualGlobal(CONFIGS);
    // This project file doesn't define configs, so it must not be used for canonical configs.
    if (configs == null) {
      return ImmutableList.of();
    }

    String sclConfigNameForMessage = sclConfigName;
    Collection<String> sclConfigValue = null;
    if (sclConfigName.isEmpty()) {
      // If there's no --scl_config, try to use the default_config.
      var defaultConfigName = (String) sclContent.getResidualGlobal(DEFAULT_CONFIG);
      try {
        sclConfigValue = validateDefaultConfig(defaultConfigName, configs);
        sclConfigNameForMessage = defaultConfigName;
      } catch (InvalidProjectFileException e) {
        throw new FlagSetFunctionException(
            new UnsupportedConfigException(
                String.format(
                    "This project's builds must set --scl_config because %s.\n%s",
                    e.getMessage(), supportedConfigsDesc(projectFile, configs))),
            Transience.PERSISTENT);
      }
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
      sclConfigValue = configs.get(sclConfigName);
    }
    validateNoExtraFlagsSet(targetOptions, userOptions, sclConfigValue);
    eventHandler.handle(
        Event.info(
            String.format(
                "Applying flags from the config '%s' defined in %s: %s ",
                sclConfigNameForMessage, projectFile, sclConfigValue)));
    return ImmutableList.copyOf(sclConfigValue);
  }

  private static Collection<String> validateDefaultConfig(
      @Nullable String defaultConfigName, Dict<String, Collection<String>> configs)
      throws InvalidProjectFileException {
    if (defaultConfigName == null) {
      throw new InvalidProjectFileException("no default_config is defined");
    }

    if (!configs.containsKey(defaultConfigName)) {
      throw new InvalidProjectFileException(
          String.format("default_config refers to a nonexistent config: %s", defaultConfigName));
    }

    return configs.get(defaultConfigName);
  }

  /**
   * Enforces that the user did not set any output-affecting options that are not present in the
   * selected config in a blazerc or on the command line. Conflicting output-affecting options may
   * be set in global RC files (including the {@code InvocationPolicy}). Flags that do not affect
   * outputs are always allowed.
   */
  private void validateNoExtraFlagsSet(
      BuildOptions targetOptions,
      ImmutableMap<String, String> userOptions,
      Collection<String> flagsFromSelectedConfig)
      throws FlagSetFunctionException {
    ImmutableList.Builder<String> allOptionsAsStringsBuilder = new ImmutableList.Builder<>();
    // All potentially conflicting user options also appear in targetOptions
    targetOptions.getStarlarkOptions().keySet().stream()
        .map(Object::toString)
        .forEach(allOptionsAsStringsBuilder::add);
    for (FragmentOptions fragmentOptions : targetOptions.getNativeOptions()) {
      if (fragmentOptions.getClass().equals(TestConfiguration.TestOptions.class)) {
        continue;
      }
      fragmentOptions.asMap().keySet().forEach(allOptionsAsStringsBuilder::add);
    }
    ImmutableList<String> allOptionsAsStrings = allOptionsAsStringsBuilder.build();
    ImmutableSet<String> overlap =
        userOptions.keySet().stream()
            // Remove options that aren't part of BuildOptions
            .filter(
                option ->
                    allOptionsAsStrings.contains(
                        Iterables.get(Splitter.on("=").split(option), 0)
                            .replaceFirst("--", "")
                            .replaceAll("'", "")))
            .filter(option -> !option.startsWith("--scl_config"))
            .filter(option -> !flagsFromSelectedConfig.contains(option))
            .map(
                option ->
                    userOptions.get(option).isEmpty()
                        ? "'" + option + "'"
                        : "'" + option + "' (expanded from '" + userOptions.get(option) + "')")
            .collect(toImmutableSet());
    // TODO(b/341930725): Allow user options if they are also part of the --scl_config.
    if (!overlap.isEmpty()) {
      throw new FlagSetFunctionException(
          new UnsupportedConfigException(
              String.format(
                  "When --enforce_project_configs is set, --scl_config must be the only"
                      + " configuration-affecting flag in the build. Found %s in the command line"
                      + " or user blazerc",
                  overlap)),
          Transience.PERSISTENT);
    }
  }

  /** Returns a user-friendly description of project-supported configurations. */
  private static String supportedConfigsDesc(
      Label projectFile, Dict<String, Collection<String>> configs) {
    String ans = "\nThis project supports:\n";
    for (var configInfo : configs.entrySet()) {
      ans += String.format("  --scl_config=%s: %s\n", configInfo.getKey(), configInfo.getValue());
    }
    ans += String.format("\nThis policy is defined in %s.\n", projectFile.toPathFragment());
    return ans;
  }

  /**
   * Converts a list of flags in string form to a set of actual flags parsed by the options parser.
   */
  @Nullable
  private static ParsedFlagsValue parseFlags(
      Collection<String> flagsAsStarlarkList, Environment env) throws InterruptedException {
    RepositoryMappingValue mainRepositoryMappingValue =
        (RepositoryMappingValue) env.getValue(RepositoryMappingValue.key(RepositoryName.MAIN));
    if (mainRepositoryMappingValue == null) {
      return null;
    }
    RepoContext mainRepoContext =
        RepoContext.of(RepositoryName.MAIN, mainRepositoryMappingValue.repositoryMapping());
    return (ParsedFlagsValue)
        env.getValue(
            ParsedFlagsValue.Key.create(
                ImmutableList.copyOf(flagsAsStarlarkList), mainRepoContext.rootPackage()));
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

  private static final class InvalidProjectFileException extends Exception {
    InvalidProjectFileException(String msg) {
      super(msg);
    }
  }
}
