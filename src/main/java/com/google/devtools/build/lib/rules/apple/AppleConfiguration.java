// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.apple;

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;

import java.util.List;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * A configuration containing flags required for Apple platforms and tools.
 */
public class AppleConfiguration extends BuildConfiguration.Fragment {
  public static final String XCODE_VERSION_ENV_NAME = "XCODE_VERSION_OVERRIDE";
  /**
   * Environment variable name for the apple SDK version. If unset, uses the system default of the
   * host for the platform in the value of {@link #APPLE_SDK_PLATFORM_ENV_NAME}.
   **/
  public static final String APPLE_SDK_VERSION_ENV_NAME = "APPLE_SDK_VERSION_OVERRIDE";
  /**
   * Environment variable name for the apple SDK platform. This should be set for all actions that
   * require an apple SDK. The valid values consist of {@link Platform} names.
   **/
  public static final String APPLE_SDK_PLATFORM_ENV_NAME = "APPLE_SDK_PLATFORM";

  private final String iosSdkVersion;
  private final String iosCpu;
  private final Optional<String> xcodeVersionOverride;
  private final List<String> iosMultiCpus;
  @Nullable private final Label defaultProvisioningProfileLabel;

  AppleConfiguration(AppleCommandLineOptions appleOptions) {
    this.iosSdkVersion = Preconditions.checkNotNull(appleOptions.iosSdkVersion, "iosSdkVersion");
    String xcodeVersionFlag = Preconditions.checkNotNull(appleOptions.xcodeVersion, "xcodeVersion");
    this.xcodeVersionOverride =
        xcodeVersionFlag.isEmpty() ? Optional.<String>absent() : Optional.of(xcodeVersionFlag);
    this.iosCpu = Preconditions.checkNotNull(appleOptions.iosCpu, "iosCpu");
    this.iosMultiCpus = Preconditions.checkNotNull(appleOptions.iosMultiCpus, "iosMultiCpus");
    this.defaultProvisioningProfileLabel = appleOptions.defaultProvisioningProfile;
  }

  /**
   * Returns the SDK version for ios SDKs (whether they be for simulator or device). This is
   * directly derived from --ios_sdk_version. Format "x.y" (for example, "6.4").
   */
  public String getIosSdkVersion() {
    return iosSdkVersion;
  }

  /**
   * Returns the value of the xcode version override build flag, if specified. This is directly
   * derived from --xcode_version_override. Format "x(.y)(.z)" (for example, "7", or "6.4",
   * or "7.0.1").
   */
  public Optional<String> getXcodeVersionOverride() {
    return xcodeVersionOverride;
  }

  /**
   * Returns a map of environment variables (derived from configuration) that should be propagated
   * for actions pertaining to building ios applications. Keys are variable names and values are
   * their corresponding values.
   */
  // TODO(bazel-team): Repurpose for non-ios platforms.
  public Map<String, String> getEnvironmentForIosAction() {
    ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();
    if (xcodeVersionOverride.isPresent()) {
      builder.put(XCODE_VERSION_ENV_NAME, xcodeVersionOverride.get());
    }
    builder.put(APPLE_SDK_VERSION_ENV_NAME, iosSdkVersion);
    builder.put(APPLE_SDK_PLATFORM_ENV_NAME, Platform.forArch(getIosCpu()).getNameInPlist());
    return builder.build();
  }
  
  public String getIosCpu() {
    return iosCpu;
  }
  
  /**
   * Returns the platform of the configuration for the current bundle, based on configured
   * architectures (for example, {@code i386} maps to {@link Platform#IOS_SIMULATOR}).
   *
   * <p>If {@link #getIosMultiCpus()} is set, returns {@link Platform#IOS_DEVICE} if any of the
   * architectures matches it, otherwise returns the mapping for {@link #getIosCpu()}.
   *
   * <p>Note that this method should not be used to determine the platform for code compilation.
   * Derive the platform from {@link #getIosCpu()} instead.
   */
  // TODO(bazel-team): This method should be enabled to return multiple values once all call sites
  // (in particular actool, bundlemerge, momc) have been upgraded to support multiple values.
  public Platform getBundlingPlatform() {
    for (String architecture : getIosMultiCpus()) {
      if (Platform.forArch(architecture) == Platform.IOS_DEVICE) {
        return Platform.IOS_DEVICE;
      }
    }
    return Platform.forArch(getIosCpu());
  }
  
  /**
   * Returns the architecture for which we keep dependencies that should be present only once (in a
   * single architecture).
   *
   * <p>When building with multiple architectures there are some dependencies we want to avoid
   * duplicating: they would show up more than once in the same location in the final application
   * bundle which is illegal. Instead we pick one architecture for which to keep all dependencies
   * and discard any others.
   */
  public String getDependencySingleArchitecture() {
    if (!getIosMultiCpus().isEmpty()) {
      return getIosMultiCpus().get(0);
    }
    return getIosCpu();
  }
  
  /**
   * List of all CPUs that this invocation is being built for. Different from {@link #getIosCpu()}
   * which is the specific CPU <b>this target</b> is being built for.
   */
  public List<String> getIosMultiCpus() {
    return iosMultiCpus;
  }

  /**
   * Returns the label of the default provisioning profile to use when bundling/signing an ios
   * application. Returns null if the target platform is not an iOS device (for example, if
   * iOS simulator is being targeted).
   */
  @Nullable public Label getDefaultProvisioningProfileLabel() {
    return defaultProvisioningProfileLabel;
  }

  /**
   * Loads {@link AppleConfiguration} from build options.
   */
  public static class Loader implements ConfigurationFragmentFactory {
    @Override
    public AppleConfiguration create(ConfigurationEnvironment env, BuildOptions buildOptions)
        throws InvalidConfigurationException {
      AppleCommandLineOptions appleOptions = buildOptions.get(AppleCommandLineOptions.class);

      return new AppleConfiguration(appleOptions);
    }

    @Override
    public Class<? extends BuildConfiguration.Fragment> creates() {
      return AppleConfiguration.class;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.<Class<? extends FragmentOptions>>of(AppleCommandLineOptions.class);
    }
  }
}
