// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.toolchains;

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.CommonOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.bazel.bzlmod.BazelDepGraphValue;
import com.google.devtools.build.lib.bazel.bzlmod.Module;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.SignedTargetPattern;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.rules.platform.PlatformRule;
import com.google.devtools.build.lib.server.FailureDetails.Analysis;
import com.google.devtools.build.lib.server.FailureDetails.Analysis.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Toolchain;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.ConfiguredValueCreationException;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.skyframe.SaneAnalysisException;
import com.google.devtools.build.lib.skyframe.TargetPatternUtil;
import com.google.devtools.build.lib.skyframe.TargetPatternUtil.InvalidTargetPatternException;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.skyframe.toolchains.PlatformLookupUtil.InvalidPlatformException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/** {@link SkyFunction} that returns all registered execution platforms available. */
public class RegisteredExecutionPlatformsFunction implements SkyFunction {

  @SerializationConstant
  static final FilteringPolicy HAS_PLATFORM_INFO =
      (Target target, boolean explicit) -> explicit || PlatformLookupUtil.hasPlatformInfo(target);

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws RegisteredExecutionPlatformsFunctionException, InterruptedException {
    RegisteredExecutionPlatformsValue.Key key = (RegisteredExecutionPlatformsValue.Key) skyKey;
    BuildConfigurationValue configuration =
        (BuildConfigurationValue) env.getValue(key.configurationKey());
    RepositoryMappingValue mainRepoMapping =
        (RepositoryMappingValue) env.getValue(RepositoryMappingValue.key(RepositoryName.MAIN));
    if (env.valuesMissing()) {
      return null;
    }

    TargetPattern.Parser mainRepoParser =
        new TargetPattern.Parser(
            PathFragment.EMPTY_FRAGMENT, RepositoryName.MAIN, mainRepoMapping.repositoryMapping());
    ImmutableList.Builder<SignedTargetPattern> targetPatternBuilder = new ImmutableList.Builder<>();

    // Get the execution platforms from the configuration.
    PlatformConfiguration platformConfiguration =
        configuration.getFragment(PlatformConfiguration.class);
    if (platformConfiguration != null) {
      try {
        targetPatternBuilder.addAll(
            TargetPatternUtil.parseAllSigned(
                platformConfiguration.getExtraExecutionPlatforms(), mainRepoParser));
      } catch (InvalidTargetPatternException e) {
        throw new RegisteredExecutionPlatformsFunctionException(
            new InvalidExecutionPlatformLabelException(e), Transience.PERSISTENT);
      }
    }

    // Get registered execution platforms from the external dep graph.
    ImmutableList<TargetPattern> bzlmodExecutionPlatforms = getBzlmodExecutionPlatforms(env);
    if (bzlmodExecutionPlatforms == null) {
      return null;
    }
    targetPatternBuilder.addAll(TargetPatternUtil.toSigned(bzlmodExecutionPlatforms));

    // Expand target patterns.
    ImmutableList<Label> platformLabels;
    try {
      platformLabels =
          TargetPatternUtil.expandTargetPatterns(
              env, targetPatternBuilder.build(), HAS_PLATFORM_INFO);
      if (env.valuesMissing()) {
        return null;
      }
    } catch (TargetPatternUtil.InvalidTargetPatternException e) {
      throw new RegisteredExecutionPlatformsFunctionException(
          new InvalidExecutionPlatformLabelException(e), Transience.PERSISTENT);
    }

    // Load the configured target for each, and get the declared execution platforms providers.
    ImmutableMap<ConfiguredTargetKey, PlatformInfo> registeredExecutionPlatforms =
        configureRegisteredExecutionPlatforms(env, configuration, platformLabels);
    if (env.valuesMissing()) {
      return null;
    }

    // Check which platforms are valid according to their configuration.
    ImmutableList.Builder<ConfiguredTargetKey> platformKeys = new ImmutableList.Builder<>();
    ImmutableMap.Builder<Label, String> rejectedPlatforms =
        key.debug() ? new ImmutableMap.Builder<>() : null;
    for (Map.Entry<ConfiguredTargetKey, PlatformInfo> entry :
        registeredExecutionPlatforms.entrySet()) {
      ConfiguredTargetKey configuredTargetKey = entry.getKey();
      PlatformInfo platformInfo = entry.getValue();

      try {
        Consumer<String> errorHandler =
            key.debug() ? message -> rejectedPlatforms.put(platformInfo.label(), message) : null;
        if (ConfigMatchingUtil.validate(
            platformInfo.label(),
            platformInfo.requiredSettings(),
            errorHandler,
            PlatformRule.REQUIRED_SETTINGS_ATTR)) {
          platformKeys.add(configuredTargetKey);
        }
      } catch (InvalidConfigurationException e) {
        throw new RegisteredExecutionPlatformsFunctionException(
            new InvalidExecutionPlatformLabelException(platformInfo.label(), e),
            Transience.PERSISTENT);
      }
    }

    return RegisteredExecutionPlatformsValue.create(
        platformKeys.build(),
        rejectedPlatforms != null ? rejectedPlatforms.buildKeepingLast() : null);
  }

  @Nullable
  private static ImmutableList<TargetPattern> getBzlmodExecutionPlatforms(Environment env)
      throws InterruptedException, RegisteredExecutionPlatformsFunctionException {
    BazelDepGraphValue bazelDepGraphValue =
        (BazelDepGraphValue) env.getValue(BazelDepGraphValue.KEY);
    if (bazelDepGraphValue == null) {
      return null;
    }
    ImmutableList.Builder<TargetPattern> executionPlatforms = ImmutableList.builder();
    for (Module module : bazelDepGraphValue.getDepGraph().values()) {
      TargetPattern.Parser parser =
          new TargetPattern.Parser(
              PathFragment.EMPTY_FRAGMENT,
              bazelDepGraphValue.getCanonicalRepoNameLookup().inverse().get(module.getKey()),
              bazelDepGraphValue.getFullRepoMapping(module.getKey()));
      for (String pattern : module.getExecutionPlatformsToRegister()) {
        try {
          executionPlatforms.add(parser.parse(pattern));
        } catch (TargetParsingException e) {
          throw new RegisteredExecutionPlatformsFunctionException(
              new InvalidExecutionPlatformLabelException(pattern, e), Transience.PERSISTENT);
        }
      }
    }
    return executionPlatforms.build();
  }

  @Nullable
  private static ImmutableMap<ConfiguredTargetKey, PlatformInfo>
      configureRegisteredExecutionPlatforms(
          Environment env, BuildConfigurationValue configuration, List<Label> labels)
          throws InterruptedException, RegisteredExecutionPlatformsFunctionException {
    ImmutableSet<ConfiguredTargetKey> keys =
        labels.stream()
            .map(
                label ->
                    ConfiguredTargetKey.builder()
                        .setLabel(label)
                        .setConfiguration(configuration)
                        .build())
            .collect(toImmutableSet());

    SkyframeLookupResult values = env.getValuesAndExceptions(keys);
    ImmutableMap.Builder<ConfiguredTargetKey, PlatformInfo> platforms =
        new ImmutableMap.Builder<>();
    boolean valuesMissing = false;
    for (ConfiguredTargetKey platformKey : keys) {
      Label platformLabel = platformKey.getLabel();
      try {
        SkyValue value = values.getOrThrow(platformKey, ConfiguredValueCreationException.class);
        if (value == null) {
          valuesMissing = true;
          continue;
        }
        ConfiguredTarget target = ((ConfiguredTargetValue) value).getConfiguredTarget();
        PlatformInfo platformInfo = PlatformProviderUtils.platform(target);
        if (platformInfo == null) {
          throw new RegisteredExecutionPlatformsFunctionException(
              new InvalidPlatformException(platformLabel), Transience.PERSISTENT);
        }

        // Update the key so that any aliases are resolved.
        platformLabel = target.getLabel();
        platformKey =
            ConfiguredTargetKey.builder()
                .setLabel(platformLabel)
                .setConfigurationKey(BuildConfigurationKey.create(CommonOptions.EMPTY_OPTIONS))
                .build();

        platforms.put(platformKey, platformInfo);
      } catch (ConfiguredValueCreationException e) {
        throw new RegisteredExecutionPlatformsFunctionException(
            new InvalidPlatformException(platformLabel, e), Transience.PERSISTENT);
      }
    }

    if (valuesMissing) {
      return null;
    }
    return platforms.buildOrThrow();
  }

  /**
   * Used to indicate that the given {@link Label} represents a {@link ConfiguredTarget} which is
   * not a valid {@link PlatformInfo} provider.
   */
  static final class InvalidExecutionPlatformLabelException extends ToolchainException
      implements SaneAnalysisException {

    InvalidExecutionPlatformLabelException(TargetPatternUtil.InvalidTargetPatternException e) {
      this(e.getInvalidPattern(), e.getTpe());
    }

    InvalidExecutionPlatformLabelException(String invalidPattern, TargetParsingException e) {
      super(
          String.format(
              "invalid registered execution platform '%s': %s", invalidPattern, e.getMessage()),
          e);
    }

    public InvalidExecutionPlatformLabelException(Label platform, InvalidConfigurationException e) {
      super(
          String.format("invalid registered execution platform '%s': %s", platform, e.getMessage()),
          e);
    }

    @Override
    protected Toolchain.Code getDetailedCode() {
      return Toolchain.Code.INVALID_PLATFORM_VALUE;
    }

    @Override
    public DetailedExitCode getDetailedExitCode() {
      return DetailedExitCode.of(
          FailureDetail.newBuilder()
              .setMessage(getMessage())
              .setAnalysis(Analysis.newBuilder().setCode(Code.INVALID_EXECUTION_PLATFORM))
              .build());
    }
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by {@link
   * #compute}.
   */
  private static class RegisteredExecutionPlatformsFunctionException extends SkyFunctionException {

    private RegisteredExecutionPlatformsFunctionException(
        InvalidExecutionPlatformLabelException cause, Transience transience) {
      super(cause, transience);
    }

    private RegisteredExecutionPlatformsFunctionException(
        InvalidPlatformException cause, Transience transience) {
      super(cause, transience);
    }
  }
}
