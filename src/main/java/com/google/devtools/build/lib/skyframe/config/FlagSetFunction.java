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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.Label.RepoContext;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.skyframe.BzlLoadFailedException;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.skyframe.config.ParsedFlagsFunction.ParsedFlagsFunctionException;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import javax.annotation.Nullable;

/**
 * A SkyFunction that, given an scl file path and the name of scl configs, does the following: 1)
 * call {@link BzlLoadFunction} to load the content of scl files given the provided scl config name
 * 2) call {@link ParsedFlagsFunction} to parse the list of options 3) define a patch transition and
 * applies the transition to the targetOptions which was used for creating topLevel configuration.
 *
 * <p>If given an unknown {@link CoreOptions.sclConfig}, {@link FlagSetFunction} will return the
 * original {@link BuildOptions} and will not error out.
 */
public class FlagSetFunction implements SkyFunction {

  @Override
  @SuppressWarnings("unchecked")
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws FlagSetFunctionException, ParsedFlagsFunctionException, InterruptedException {
    FlagSetValue.Key key = (FlagSetValue.Key) skyKey.argument();

    if (key.getProjectFile() == null || key.getSclConfig().isEmpty()) {
      return FlagSetValue.create(key.getTargetOptions());
    }

    BzlLoadValue sclLoadValue = loadSclFile(key.getProjectFile(), env);

    if (sclLoadValue == null) {
      return null;
    }

    RepositoryMappingValue mainRepositoryMappingValue =
        (RepositoryMappingValue) env.getValue(RepositoryMappingValue.key(RepositoryName.MAIN));
    if (mainRepositoryMappingValue == null) {
      return null;
    }

    RepoContext mainRepoContext =
        RepoContext.of(RepositoryName.MAIN, mainRepositoryMappingValue.getRepositoryMapping());

    List<String> rawFlags = new ArrayList<>();
    if (sclLoadValue.getModule().getGlobal(key.getSclConfig()) != null) {
      rawFlags.addAll(
          (Collection<? extends String>) sclLoadValue.getModule().getGlobal(key.getSclConfig()));
    } else {
      return FlagSetValue.create(key.getTargetOptions());
    }

    ParsedFlagsValue parsedFlagsValue;
    try {
      parsedFlagsValue =
          (ParsedFlagsValue)
              env.getValueOrThrow(
                  ParsedFlagsValue.Key.create(
                      ImmutableList.copyOf(rawFlags), mainRepoContext.rootPackage()),
                  ParsedFlagsFunctionException.class);
    } catch (ParsedFlagsFunctionException e) {
      throw new FlagSetFunctionException(e, Transience.PERSISTENT);
    }

    if (parsedFlagsValue == null) {
      return null;
    }

    BuildOptions adjustedBuildOptions;
    try {
      OptionsParsingResult optionsParsingResult = parsedFlagsValue.flags().parse();
      FlagSetTransition transition = new FlagSetTransition(optionsParsingResult);
      BuildOptionsView buildOptionsView =
          new BuildOptionsView(key.getTargetOptions(), parsedFlagsValue.flags().optionsClasses());
      adjustedBuildOptions =
          Iterables.getOnlyElement(transition.apply(buildOptionsView, env.getListener()).values());
    } catch (OptionsParsingException e) {
      throw new FlagSetFunctionException(e, Transience.PERSISTENT);
    }

    return FlagSetValue.create(adjustedBuildOptions);
  }

  private BzlLoadValue loadSclFile(Label sclFileLabel, Environment env)
      throws FlagSetFunctionException, InterruptedException {
    BzlLoadValue bzlLoadValue;
    try {
      bzlLoadValue =
          (BzlLoadValue)
              env.getValueOrThrow(
                  BzlLoadValue.keyForBuild(sclFileLabel), BzlLoadFailedException.class);
    } catch (BzlLoadFailedException e) {
      throw new FlagSetFunctionException(e, Transience.PERSISTENT);
    }
    return bzlLoadValue;
  }

  private static final class FlagSetFunctionException extends SkyFunctionException {
    FlagSetFunctionException(Exception cause, Transience transience) {
      super(cause, transience);
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
