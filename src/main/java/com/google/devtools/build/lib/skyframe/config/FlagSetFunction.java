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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.cmdline.Label.RepoContext;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.skyframe.ProjectValue;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.skyframe.config.ParsedFlagsFunction.ParsedFlagsFunctionException;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
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
        getSclConfig(projectValue, key.getSclConfig(), key.enforceCanonical());
    ParsedFlagsValue parsedFlags = parseFlags(sclConfigAsStarlarkList, env);
    if (parsedFlags == null) {
      return null;
    }
    return FlagSetValue.create(changeOptions(key.getTargetOptions(), parsedFlags, env));
  }

  /**
   * Given an .scl file and {@code --scl_config} value, returns the flags denoted by that {@code
   * --scl_config}. Flags are a list of strings (not parsed through the options parser).
   */
  @SuppressWarnings("unchecked")
  private ImmutableList<String> getSclConfig(
      ProjectValue sclContent, String sclConfigName, boolean enforceCanonical)
      throws FlagSetFunctionException {
    var configs = (Dict<String, Collection<String>>) sclContent.getResidualGlobal(CONFIGS);
    var sclConfigValue = configs.get(sclConfigName);
    var supportedConfigs =
        (Dict<String, Collection<String>>) sclContent.getResidualGlobal(SUPPORTED_CONFIGS);

    // Look for invalid use cases.
    String errorMsg = null;
    if (!enforceCanonical) {
      // Calling code already handled non-existent --scl_config values and !enforceCanonical.
      Preconditions.checkNotNull(sclConfigValue);
    } else if (sclConfigName.isEmpty()) {
      if (supportedConfigs == null) {
        // No --scl_config but no project-declared supported configs. This is a valid use case.
        return ImmutableList.of();
      }
      errorMsg =
          String.format(
              "--scl_config not set. Must be one of [%s]",
              String.join(",", supportedConfigs.keySet()));
    } else if (sclConfigValue == null) {
      if (supportedConfigs == null) {
        // Bad --scl_config but no project-declared supported configs. No-op just like in
        // --noenforce_canonical_configs mode.
        return ImmutableList.of();
      }
      errorMsg =
          String.format(
              "--scl_config=%s not found. Must be one of [%s]",
              sclConfigName, String.join(",", supportedConfigs.keySet()));
    } else if (supportedConfigs != null && !supportedConfigs.containsKey(sclConfigName)) {
      errorMsg =
          String.format(
              "--scl_config=%s unsupported. Must be one of [%s]",
              sclConfigName, String.join(",", supportedConfigs.keySet()));
    }

    // Error out or return the flags to set.
    if (errorMsg != null) {
      throw new FlagSetFunctionException(
          new UnsupportedConfigException(errorMsg), Transience.PERSISTENT);
    }
    return ImmutableList.copyOf(sclConfigValue);
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
  private BuildOptions changeOptions(
      BuildOptions fromOptions, ParsedFlagsValue parsedFlags, Environment env)
      throws FlagSetFunctionException, InterruptedException {
    try {
      FlagSetTransition transition = new FlagSetTransition(parsedFlags.flags().parse());
      BuildOptionsView buildOptionsView =
          new BuildOptionsView(fromOptions, parsedFlags.flags().optionsClasses());
      return Iterables.getOnlyElement(
          transition.apply(buildOptionsView, env.getListener()).values());
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

  /** Transition that applies the config defines in PROJECT.scl to existing buildOptions */
  private static class FlagSetTransition implements PatchTransition {
    public final OptionsParsingResult parsingResult;

    public FlagSetTransition(OptionsParsingResult parsingResult) {
      this.parsingResult = parsingResult;
    }

    @Override
    public BuildOptions patch(BuildOptionsView originalOptions, EventHandler eventHandler) {
      BuildOptions toOptions = originalOptions.underlying().clone();
      return toOptions.applyParsingResult(parsingResult);
    }
  }
}
