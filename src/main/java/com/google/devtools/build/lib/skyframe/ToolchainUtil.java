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

import com.google.auto.value.AutoValue;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.ToolchainContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetFunction.ConfiguredValueCreationException;
import com.google.devtools.build.lib.skyframe.RegisteredToolchainsFunction.InvalidToolchainLabelException;
import com.google.devtools.build.lib.skyframe.ToolchainResolutionFunction.NoToolchainFoundException;
import com.google.devtools.build.lib.skyframe.ToolchainResolutionValue.ToolchainResolutionKey;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.ValueOrException;
import com.google.devtools.build.skyframe.ValueOrException4;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Common code to create a {@link ToolchainContext} given a set of required toolchain type labels.
 */
public class ToolchainUtil {

  /**
   * Returns a new {@link ToolchainContext}, with the correct toolchain labels based on the results
   * of the {@link ToolchainResolutionFunction}.
   */
  @Nullable
  public static ToolchainContext createToolchainContext(
      Environment env,
      String targetDescription,
      Set<Label> requiredToolchains,
      @Nullable BuildConfiguration configuration)
      throws ToolchainContextException, InterruptedException {

    // In some cases this is called with a missing configuration, so we skip toolchain context.
    if (configuration == null) {
      return null;
    }

    // TODO(katre): Load several possible execution platforms, and select one based on available
    // toolchains.

    // Load the host and target platforms for the current configuration.
    PlatformDescriptors platforms = loadPlatformDescriptors(env, configuration);
    if (platforms == null) {
      return null;
    }

    // TODO(katre): This will change with remote execution.
    PlatformInfo executionPlatform = platforms.hostPlatform();
    PlatformInfo targetPlatform = platforms.targetPlatform();

    ImmutableBiMap<Label, Label> resolvedLabels =
        resolveToolchainLabels(
            env, requiredToolchains, configuration, executionPlatform, targetPlatform);
    if (resolvedLabels == null) {
      return null;
    }

    ToolchainContext toolchainContext =
        ToolchainContext.create(
            targetDescription,
            executionPlatform,
            targetPlatform,
            requiredToolchains,
            resolvedLabels);
    return toolchainContext;
  }

  /**
   * Data class to hold platform descriptors loaded based on the current {@link BuildConfiguration}.
   */
  @AutoValue
  protected abstract static class PlatformDescriptors {
    abstract PlatformInfo hostPlatform();

    abstract PlatformInfo targetPlatform();

    protected static PlatformDescriptors create(
        PlatformInfo hostPlatform, PlatformInfo targetPlatform) {
      return new AutoValue_ToolchainUtil_PlatformDescriptors(hostPlatform, targetPlatform);
    }
  }

  /**
   * Returns the {@link PlatformInfo} provider from the {@link ConfiguredTarget} in the {@link
   * ValueOrException}, or {@code null} if the {@link ConfiguredTarget} is not present. If the
   * {@link ConfiguredTarget} does not have a {@link PlatformInfo} provider, a {@link
   * InvalidPlatformException} is thrown, wrapped in a {@link ToolchainContextException}.
   */
  @Nullable
  private static PlatformInfo findPlatformInfo(
      ValueOrException<ConfiguredValueCreationException> valueOrException,
      String platformType,
      Environment env)
      throws ConfiguredValueCreationException, ToolchainContextException {

    ConfiguredTargetValue ctv = (ConfiguredTargetValue) valueOrException.get();
    if (ctv == null) {
      return null;
    }

    ConfiguredTarget configuredTarget = ctv.getConfiguredTarget();
    Target target = null;
    try {
      target =
          ((PackageValue)
                  env.getValue(
                      PackageValue.key(configuredTarget.getLabel().getPackageIdentifier())))
              .getPackage()
              .getTarget(configuredTarget.getLabel().getName());
    } catch (NoSuchTargetException | InterruptedException e) {
      throw new IllegalStateException("Unable to get Target when computing PlatformInfo.", e);
    }
    PlatformInfo platformInfo = PlatformProviderUtils.platform(configuredTarget);
    if (platformInfo == null) {
      throw new ToolchainContextException(new InvalidPlatformException(platformType, target));
    }

    return platformInfo;
  }

  @Nullable
  private static PlatformDescriptors loadPlatformDescriptors(
      Environment env, BuildConfiguration configuration)
      throws InterruptedException, ToolchainContextException {
    PlatformConfiguration platformConfiguration =
        configuration.getFragment(PlatformConfiguration.class);
    if (platformConfiguration == null) {
      return null;
    }
    Label hostPlatformLabel = platformConfiguration.getHostPlatform();
    Label targetPlatformLabel = platformConfiguration.getTargetPlatforms().get(0);

    SkyKey hostPlatformKey = ConfiguredTargetKey.of(hostPlatformLabel, configuration);
    SkyKey targetPlatformKey = ConfiguredTargetKey.of(targetPlatformLabel, configuration);

    Map<SkyKey, ValueOrException<ConfiguredValueCreationException>> values =
        env.getValuesOrThrow(
            ImmutableList.of(hostPlatformKey, targetPlatformKey),
            ConfiguredValueCreationException.class);
    boolean valuesMissing = env.valuesMissing();
    try {
      PlatformInfo hostPlatform =
          findPlatformInfo(values.get(hostPlatformKey), "host platform", env);
      PlatformInfo targetPlatform =
          findPlatformInfo(values.get(targetPlatformKey), "target platform", env);

      if (valuesMissing) {
        return null;
      }

      return PlatformDescriptors.create(hostPlatform, targetPlatform);
    } catch (ConfiguredValueCreationException e) {
      throw new ToolchainContextException(e);
    }
  }

  @Nullable
  private static ImmutableBiMap<Label, Label> resolveToolchainLabels(
      Environment env,
      Set<Label> requiredToolchains,
      BuildConfiguration configuration,
      PlatformInfo executionPlatform,
      PlatformInfo targetPlatform)
      throws InterruptedException, ToolchainContextException {

    // If there are no required toolchains, bail out early.
    if (requiredToolchains.isEmpty()) {
      return ImmutableBiMap.of();
    }

    // Find the toolchains for the required toolchain types.
    List<SkyKey> registeredToolchainKeys = new ArrayList<>();
    for (Label toolchainType : requiredToolchains) {
      registeredToolchainKeys.add(
          ToolchainResolutionValue.key(
              configuration, toolchainType, targetPlatform, executionPlatform));
    }

    Map<
            SkyKey,
            ValueOrException4<
                NoToolchainFoundException, ConfiguredValueCreationException,
                InvalidToolchainLabelException, EvalException>>
        results =
            env.getValuesOrThrow(
                registeredToolchainKeys,
                NoToolchainFoundException.class,
                ConfiguredValueCreationException.class,
                InvalidToolchainLabelException.class,
                EvalException.class);
    boolean valuesMissing = false;

    // Load the toolchains.
    ImmutableBiMap.Builder<Label, Label> builder = new ImmutableBiMap.Builder<>();
    List<Label> missingToolchains = new ArrayList<>();
    for (Map.Entry<
            SkyKey,
            ValueOrException4<
                NoToolchainFoundException, ConfiguredValueCreationException,
                InvalidToolchainLabelException, EvalException>>
        entry : results.entrySet()) {
      try {
        Label requiredToolchainType =
            ((ToolchainResolutionKey) entry.getKey().argument()).toolchainType();
        ValueOrException4<
                NoToolchainFoundException, ConfiguredValueCreationException,
                InvalidToolchainLabelException, EvalException>
            valueOrException = entry.getValue();
        if (valueOrException.get() == null) {
          valuesMissing = true;
        } else {
          Label toolchainLabel =
              ((ToolchainResolutionValue) valueOrException.get()).toolchainLabel();
          builder.put(requiredToolchainType, toolchainLabel);
        }
      } catch (NoToolchainFoundException e) {
        // Save the missing type and continue looping to check for more.
        missingToolchains.add(e.missingToolchainType());
      } catch (ConfiguredValueCreationException e) {
        throw new ToolchainContextException(e);
      } catch (InvalidToolchainLabelException e) {
        throw new ToolchainContextException(e);
      } catch (EvalException e) {
        throw new ToolchainContextException(e);
      }
    }

    if (!missingToolchains.isEmpty()) {
      throw new ToolchainContextException(new UnresolvedToolchainsException(missingToolchains));
    }

    if (valuesMissing) {
      return null;
    }

    return builder.build();
  }

  /** Exception used when a platform label is not a valid platform. */
  public static final class InvalidPlatformException extends Exception {
    public InvalidPlatformException(String platformType, Target target) {
      super(
          String.format(
              "Target %s was found as the %s, but does not provide PlatformInfo",
              target, platformType));
    }
  }

  /** Exception used when a toolchain type is required but no matching toolchain is found. */
  public static final class UnresolvedToolchainsException extends Exception {
    private final ImmutableList<Label> missingToolchainTypes;

    public UnresolvedToolchainsException(List<Label> missingToolchainTypes) {
      super(
          String.format(
              "no matching toolchains found for types %s",
              Joiner.on(", ").join(missingToolchainTypes)));
      this.missingToolchainTypes = ImmutableList.copyOf(missingToolchainTypes);
    }

    public ImmutableList<Label> missingToolchainTypes() {
      return missingToolchainTypes;
    }
  }

  /** Exception used to wrap exceptions during toolchain resolution. */
  public static class ToolchainContextException extends Exception {
    public ToolchainContextException(InvalidPlatformException e) {
      super(e);
    }

    public ToolchainContextException(UnresolvedToolchainsException e) {
      super(e);
    }

    public ToolchainContextException(ConfiguredValueCreationException e) {
      super(e);
    }

    public ToolchainContextException(InvalidToolchainLabelException e) {
      super(e);
    }

    public ToolchainContextException(EvalException e) {
      super(e);
    }
  }
}
