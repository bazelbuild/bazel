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

package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Package;
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

/** {@link SkyFunction} that returns all registered execution platforms available. */
public class RegisteredExecutionPlatformsFunction implements SkyFunction {

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {

    BuildConfiguration configuration = (BuildConfiguration) skyKey.argument();

    ImmutableList.Builder<Label> registeredExecutionPlatformLabels = new ImmutableList.Builder<>();

    // Get the execution platforms from the configuration.
    PlatformConfiguration platformConfiguration =
        configuration.getFragment(PlatformConfiguration.class);
    registeredExecutionPlatformLabels.addAll(platformConfiguration.getExtraExecutionPlatforms());

    // Get the registered execution platforms from the WORKSPACE.
    List<Label> workspaceExecutionPlatforms = getWorkspaceExecutionPlatforms(env);
    if (workspaceExecutionPlatforms == null) {
      return null;
    }
    registeredExecutionPlatformLabels.addAll(workspaceExecutionPlatforms);

    // Load the configured target for each, and get the declared execution platforms providers.
    ImmutableList<PlatformInfo> registeredExecutionPlatforms =
        configureRegisteredExecutionPlatforms(
            env, configuration, registeredExecutionPlatformLabels.build());
    if (env.valuesMissing()) {
      return null;
    }

    return RegisteredExecutionPlatformsValue.create(registeredExecutionPlatforms);
  }

  /**
   * Loads the external package and then returns the registered execution platform labels.
   *
   * @param env the environment to use for lookups
   */
  @Nullable
  private List<Label> getWorkspaceExecutionPlatforms(Environment env) throws InterruptedException {
    PackageValue externalPackageValue =
        (PackageValue) env.getValue(PackageValue.key(Label.EXTERNAL_PACKAGE_IDENTIFIER));
    if (externalPackageValue == null) {
      return null;
    }

    Package externalPackage = externalPackageValue.getPackage();
    return externalPackage.getRegisteredExecutionPlatformLabels();
  }

  private ImmutableList<PlatformInfo> configureRegisteredExecutionPlatforms(
      Environment env, BuildConfiguration configuration, List<Label> labels)
      throws InterruptedException, RegisteredExecutionPlatformsFunctionException {
    ImmutableList<SkyKey> keys =
        labels
            .stream()
            .map(
                label ->
                    LegacySkyKey.create(
                        SkyFunctions.CONFIGURED_TARGET,
                        ConfiguredTargetKey.of(label, configuration)))
            .collect(ImmutableList.toImmutableList());

    Map<SkyKey, ValueOrException<ConfiguredValueCreationException>> values =
        env.getValuesOrThrow(keys, ConfiguredValueCreationException.class);
    ImmutableList.Builder<PlatformInfo> platforms = new ImmutableList.Builder<>();
    boolean valuesMissing = false;
    for (SkyKey key : keys) {
      ConfiguredTargetKey configuredTargetKey = (ConfiguredTargetKey) key.argument();
      Label platformLabel = configuredTargetKey.getLabel();
      try {
        ValueOrException<ConfiguredValueCreationException> valueOrException = values.get(key);
        if (valueOrException.get() == null) {
          valuesMissing = true;
          continue;
        }
        ConfiguredTarget target =
            ((ConfiguredTargetValue) valueOrException.get()).getConfiguredTarget();
        PlatformInfo platformInfo = PlatformProviderUtils.platform(target);

        if (platformInfo == null) {
          throw new RegisteredExecutionPlatformsFunctionException(
              new InvalidExecutionPlatformLabelException(platformLabel), Transience.PERSISTENT);
        }
        platforms.add(platformInfo);
      } catch (ConfiguredValueCreationException e) {
        throw new RegisteredExecutionPlatformsFunctionException(
            new InvalidExecutionPlatformLabelException(platformLabel, e), Transience.PERSISTENT);
      }
    }

    if (valuesMissing) {
      return null;
    }
    return platforms.build();
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Used to indicate that the given {@link Label} represents a {@link ConfiguredTarget} which is
   * not a valid {@link PlatformInfo} provider.
   */
  public static final class InvalidExecutionPlatformLabelException extends Exception {

    private final Label invalidLabel;

    public InvalidExecutionPlatformLabelException(Label invalidLabel) {
      super(
          String.format(
              "invalid registered execution platform '%s': "
                  + "target does not provide the PlatformInfo provider",
              invalidLabel));
      this.invalidLabel = invalidLabel;
    }

    public InvalidExecutionPlatformLabelException(
        Label invalidLabel, ConfiguredValueCreationException e) {
      super(
          String.format(
              "invalid registered execution platform '%s': %s", invalidLabel, e.getMessage()),
          e);
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
  public static class RegisteredExecutionPlatformsFunctionException extends SkyFunctionException {

    public RegisteredExecutionPlatformsFunctionException(
        InvalidExecutionPlatformLabelException cause, Transience transience) {
      super(cause, transience);
    }
  }
}
