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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.Label.RepoContext;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.ProjectValue;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.skyframe.config.ParsedFlagsFunction.ParsedFlagsFunctionException;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Collection;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;

/**
 * A SkyFunction that, given an scl file path and the name of scl configs, does the following:
 *
 * <ol>
 *   <li>calls {@link ProjectFunction} to load the content of scl files given the provided scl
 *       config name
 *   <li>calls {@link ParsedFlagsFunction} to parse the list of options
 *   <li>defines a patch transition and applies the transition to the input {@link BuildOptions}
 * </ol>
 *
 * <p>If given an unknown {@link CoreOptions.sclConfig}, {@link FlagSetFunction} returns the
 * original {@link BuildOptions} and doesn't error out.
 */
public class FlagSetFunction implements SkyFunction {
  private static final String CONFIGS = "configs";

  private static final String SUPPORTED_CONFIGS = "supported_configs";

  @Override
  @SuppressWarnings("unchecked")
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws FlagSetFunctionException, ParsedFlagsFunctionException, InterruptedException {
    FlagSetValue.Key key = (FlagSetValue.Key) skyKey.argument();
    if (key.getSclConfig().isEmpty() && !key.enforceCanonical()) {
      // No special config specified. Nothing to do.
      return FlagSetValue.create(key.getTargetOptions());
    }
    ProjectValue projectValue =
        (ProjectValue) env.getValue(new ProjectValue.Key(key.getProjectFile()));
    if (projectValue == null) {
      return null;
    }
    var configs = (Dict<String, Collection<String>>) projectValue.getResidualGlobal(CONFIGS);
    if (!key.enforceCanonical() && (configs == null || configs.get(key.getSclConfig()) == null)) {
      // If canonical configs aren't enforced, unknown --scl_configs are just no-ops. Same if the
      // `configs = {...}` variable isn't present.
      // If canonical configs are enforced, --scl_config must match some project-approved config.
      // TODO: blaze-configurability-team - Fail bad configs even without canonical enforcement?
      return FlagSetValue.create(key.getTargetOptions());
    }
    ImmutableList<String> sclConfigAsStarlarkList =
        getSclConfig(
            key.getProjectFile(), projectValue, key.getSclConfig(), key.enforceCanonical());
    ParsedFlagsValue parsedFlags = parseFlags(sclConfigAsStarlarkList, env);
    if (parsedFlags == null) {
      return null;
    }
    return FlagSetValue.create(changeOptions(key.getTargetOptions(), parsedFlags));
  }

  /**
   * Given an .scl file and {@code --scl_config} value, returns the flags denoted by that {@code
   * --scl_config}. Flags are a list of strings (not parsed through the options parser).
   */
  @SuppressWarnings("unchecked")
  private ImmutableList<String> getSclConfig(
      Label projectFile, ProjectValue sclContent, String sclConfigName, boolean enforceCanonical)
      throws FlagSetFunctionException {
    var configs = (Dict<String, Collection<String>>) sclContent.getResidualGlobal(CONFIGS);
    var sclConfigValue = configs.get(sclConfigName);
    var supportedConfigs = (Dict<String, String>) sclContent.getResidualGlobal(SUPPORTED_CONFIGS);

    // Look for invalid use cases.
    if (!enforceCanonical) {
      // Calling code already handled non-existent --scl_config values and !enforceCanonical.
      Preconditions.checkNotNull(sclConfigValue);
    } else if (supportedConfigs == null) {
      // This project doesn't declare supported configs. Allow any --scl_config just as if
      // --enforce_project_configs isn't set. This also means --scl_config=<name doesn't resolve>
      // is silently consider a no-op.
      return sclConfigValue == null ? ImmutableList.of() : ImmutableList.copyOf(sclConfigValue);
    } else if (sclConfigName.isEmpty()) {
      // This project declares supported configs and user didn't specify any config.
      throw new FlagSetFunctionException(
          new UnsupportedConfigException(
              String.format(
                  "This project's builds must set --scl_config.%s",
                  supportedConfigsDesc(projectFile, supportedConfigs))),
          Transience.PERSISTENT);
    } else if (!supportedConfigs.containsKey(sclConfigName)) {
      // This project declares supported configs and user set --scl_config to an unsupported config.
      throw new FlagSetFunctionException(
          new UnsupportedConfigException(
              String.format(
                  "--scl_config=%s is not a valid configuration for this project.%s",
                  sclConfigName, supportedConfigsDesc(projectFile, supportedConfigs))),
          Transience.PERSISTENT);
      }

    return ImmutableList.copyOf(sclConfigValue);
  }

  /** Returns a user-friendly description of project-supported configurations. */
  private static String supportedConfigsDesc(
      Label projectFile, Dict<String, String> supportedConfigs) {
    String ans = "\nThis project supports:\n";
    for (var configInfo : supportedConfigs.entrySet()) {
      ans += String.format("  --scl_config=%s: %s\n", configInfo.getKey(), configInfo.getValue());
    }
    ans += String.format("\nThis policy is defined in %s.\n", projectFile.toPathFragment());
    return ans;
  }

  /**
   * Converts a list of flags in string form to a set of actual flags parsed by the options parser.
   */
  @Nullable
  private ParsedFlagsValue parseFlags(Collection<String> flagsAsStarlarkList, Environment env)
      throws FlagSetFunctionException, InterruptedException {
    RepositoryMappingValue mainRepositoryMappingValue =
        (RepositoryMappingValue) env.getValue(RepositoryMappingValue.key(RepositoryName.MAIN));
    if (mainRepositoryMappingValue == null) {
      return null;
    }
    RepoContext mainRepoContext =
        RepoContext.of(RepositoryName.MAIN, mainRepositoryMappingValue.getRepositoryMapping());
    try {
      return (ParsedFlagsValue)
          env.getValueOrThrow(
              ParsedFlagsValue.Key.create(
                  ImmutableList.copyOf(flagsAsStarlarkList), mainRepoContext.rootPackage()),
              ParsedFlagsFunctionException.class);
    } catch (ParsedFlagsFunctionException e) {
      throw new FlagSetFunctionException(e, Transience.PERSISTENT);
    }
  }

  /** Modifies input build options with the desired flag set and returns the result. */
  private BuildOptions changeOptions(BuildOptions fromOptions, ParsedFlagsValue parsedFlags)
      throws FlagSetFunctionException {
    try {
      return parsedFlags.flags().mergeWith(fromOptions);
    } catch (OptionsParsingException e) {
      throw new FlagSetFunctionException(e, Transience.PERSISTENT);
    }
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
