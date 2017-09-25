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

package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.platform.DeclaredToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.ExternalPackageUtil;
import com.google.devtools.build.lib.rules.ExternalPackageUtil.ExternalPackageException;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetFunction.ConfiguredValueCreationException;
import com.google.devtools.build.skyframe.LegacySkyKey;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * {@link SkyFunction} that returns all registered toolchains available for toolchain resolution.
 */
public class RegisteredToolchainsFunction implements SkyFunction {

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {

    BuildConfiguration configuration = (BuildConfiguration) skyKey.argument();

    ImmutableList.Builder<Label> registeredToolchainLabels = new ImmutableList.Builder<>();

    // Get the toolchains from the configuration.
    PlatformConfiguration platformConfiguration =
        configuration.getFragment(PlatformConfiguration.class);
    registeredToolchainLabels.addAll(platformConfiguration.getExtraToolchains());

    // Get the registered toolchains from the WORKSPACE.
    registeredToolchainLabels.addAll(getWorkspaceToolchains(env));
    if (env.valuesMissing()) {
      return null;
    }

    // Load the configured target for each, and get the declared toolchain providers.
    ImmutableList<DeclaredToolchainInfo> registeredToolchains =
        configureRegisteredToolchains(env, configuration, registeredToolchainLabels.build());
    if (env.valuesMissing()) {
      return null;
    }

    return RegisteredToolchainsValue.create(registeredToolchains);
  }

  private Iterable<? extends Label> getWorkspaceToolchains(Environment env)
      throws ExternalPackageException, InterruptedException {
    List<Label> labels = ExternalPackageUtil.getRegisteredToolchainLabels(env);
    if (labels == null) {
      return ImmutableList.of();
    }
    return labels;
  }

  private ImmutableList<DeclaredToolchainInfo> configureRegisteredToolchains(
      Environment env, BuildConfiguration configuration, List<Label> labels)
      throws InterruptedException, RegisteredToolchainsFunctionException {
    ImmutableList<SkyKey> keys =
        labels
            .stream()
            .map(
                label ->
                    LegacySkyKey.create(
                        SkyFunctions.CONFIGURED_TARGET,
                        new ConfiguredTargetKey(label, configuration)))
            .collect(ImmutableList.toImmutableList());

    Map<SkyKey, ValueOrException<ConfiguredValueCreationException>> values =
        env.getValuesOrThrow(keys, ConfiguredValueCreationException.class);
    ImmutableList.Builder<DeclaredToolchainInfo> toolchains = new ImmutableList.Builder<>();
    boolean valuesMissing = false;
    for (SkyKey key : keys) {
      ConfiguredTargetKey configuredTargetKey = (ConfiguredTargetKey) key.argument();
      Label toolchainLabel = configuredTargetKey.getLabel();
      try {
        ValueOrException<ConfiguredValueCreationException> valueOrException = values.get(key);
        if (valueOrException.get() == null) {
          valuesMissing = true;
          continue;
        }
        ConfiguredTarget target =
            ((ConfiguredTargetValue) valueOrException.get()).getConfiguredTarget();
        DeclaredToolchainInfo toolchainInfo = target.getProvider(DeclaredToolchainInfo.class);

        if (toolchainInfo == null) {
          throw new RegisteredToolchainsFunctionException(
              new InvalidToolchainLabelException(toolchainLabel), Transience.PERSISTENT);
        }
        toolchains.add(toolchainInfo);
      } catch (ConfiguredValueCreationException e) {
        throw new RegisteredToolchainsFunctionException(
            new InvalidToolchainLabelException(toolchainLabel, e), Transience.PERSISTENT);
      }
    }

    if (valuesMissing) {
      return null;
    }
    return toolchains.build();
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Used to indicate that the given {@link Label} represents a {@link ConfiguredTarget} which is
   * not a valid {@link DeclaredToolchainInfo} provider.
   */
  public static final class InvalidToolchainLabelException extends Exception {

    private final Label invalidLabel;

    public InvalidToolchainLabelException(Label invalidLabel) {
      super(
          String.format(
              "invalid registered toolchain '%s': "
                  + "target does not provide the DeclaredToolchainInfo provider",
              invalidLabel));
      this.invalidLabel = invalidLabel;
    }

    public InvalidToolchainLabelException(Label invalidLabel, ConfiguredValueCreationException e) {
      super(
          String.format("invalid registered toolchain '%s': %s", invalidLabel, e.getMessage()), e);
      this.invalidLabel = invalidLabel;
    }

    public Label getInvalidLabel() {
      return invalidLabel;
    }
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by {@link
   * #compute}.
   */
  public static class RegisteredToolchainsFunctionException extends SkyFunctionException {

    public RegisteredToolchainsFunctionException(
        InvalidToolchainLabelException cause, Transience transience) {
      super(cause, transience);
    }
  }
}
