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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions.AppleBitcodeMode;
import com.google.devtools.build.lib.util.Preconditions;

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

  private static final DottedVersion MINIMUM_BITCODE_XCODE_VERSION = DottedVersion.fromString("7");

  private final DottedVersion iosSdkVersion;
  private final String iosCpu;
  private final Optional<DottedVersion> xcodeVersion;
  private final List<String> iosMultiCpus;
  private final AppleBitcodeMode bitcodeMode;
  private final Label xcodeConfigLabel;
  @Nullable private final Label defaultProvisioningProfileLabel;

  AppleConfiguration(AppleCommandLineOptions appleOptions,
      Optional<DottedVersion> xcodeVersionOverride) {
    this.iosSdkVersion = Preconditions.checkNotNull(appleOptions.iosSdkVersion, "iosSdkVersion");
    this.xcodeVersion = Preconditions.checkNotNull(xcodeVersionOverride);
    this.iosCpu = Preconditions.checkNotNull(appleOptions.iosCpu, "iosCpu");
    this.iosMultiCpus = Preconditions.checkNotNull(appleOptions.iosMultiCpus, "iosMultiCpus");
    this.bitcodeMode = appleOptions.appleBitcodeMode;
    this.xcodeConfigLabel =
        Preconditions.checkNotNull(appleOptions.xcodeVersionConfig, "xcodeConfigLabel");
    this.defaultProvisioningProfileLabel = appleOptions.defaultProvisioningProfile;
  }

  /**
   * Returns the SDK version for ios SDKs (whether they be for simulator or device). This is
   * directly derived from --ios_sdk_version. Format "x.y" (for example, "6.4").
   */
  public DottedVersion getIosSdkVersion() {
    return iosSdkVersion;
  }

  /**
   * Returns the value of the xcode version, if available. This is determined based on a combination
   * of the {@code --xcode_version} build flag and the {@code xcode_config} target defined in the
   * {@code --xcode_version_config} flag.
   */
  public Optional<DottedVersion> getXcodeVersion() {
    return xcodeVersion;
  }

  /**
   * Returns a map of environment variables (derived from configuration) that should be propagated
   * for actions pertaining to building ios applications. Keys are variable names and values are
   * their corresponding values.
   */
  // TODO(bazel-team): Repurpose for non-ios platforms.
  public Map<String, String> getEnvironmentForIosAction() {
    ImmutableMap.Builder<String, String> mapBuilder = ImmutableMap.builder();
    mapBuilder.putAll(appleTargetPlatformEnv(Platform.forIosArch(getIosCpu())));
    return mapBuilder.build();
  }
  
  /**
   * Returns a map of environment variables that should be propagated for actions that build on an
   * apple host system. These environment variables are needed by the apple toolchain. Keys are
   * variable names and values are their corresponding values.
   */
  public Map<String, String> getAppleHostSystemEnv() {
    Optional<DottedVersion> xcodeVersion = getXcodeVersion();
    ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();
    if (xcodeVersion.isPresent()) {
      builder.put(AppleConfiguration.XCODE_VERSION_ENV_NAME, xcodeVersion.get().toString());
    }
    return builder.build();   
  }

  /**
   * Returns a map of environment variables (derived from configuration) that should be propagated
   * for actions pertaining to building applications for apple platforms. These environment
   * variables are needed to use apple toolkits. Keys are variable names and values are their
   * corresponding values.
   */
  public Map<String, String> appleTargetPlatformEnv(Platform platform) {
    ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();

    // TODO(bazel-team): Handle non-ios platforms.
    if (platform == Platform.IOS_DEVICE || platform == Platform.IOS_SIMULATOR) {
      builder.put(AppleConfiguration.APPLE_SDK_VERSION_ENV_NAME, getIosSdkVersion().toString())
          .put(AppleConfiguration.APPLE_SDK_PLATFORM_ENV_NAME, platform.getNameInPlist());
    }
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
      if (Platform.forIosArch(architecture) == Platform.IOS_DEVICE) {
        return Platform.IOS_DEVICE;
      }
    }
    return Platform.forIosArch(getIosCpu());
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
   * Returns the bitcode mode to use for compilation steps. Users can control bitcode
   * mode using the {@code apple_bitcode} build flag.
   * 
   * @see AppleBitcodeMode
   */
  public AppleBitcodeMode getBitcodeMode() {
    return bitcodeMode;
  }

  /**
   * Returns the label of the xcode_config rule to use for resolving the host system xcode version.
   */
  public Label getXcodeConfigLabel() {
    return xcodeConfigLabel;
  }

  /**
   * Loads {@link AppleConfiguration} from build options.
   */
  public static class Loader implements ConfigurationFragmentFactory {
    @Override
    public AppleConfiguration create(ConfigurationEnvironment env, BuildOptions buildOptions)
        throws InvalidConfigurationException {
      AppleCommandLineOptions appleOptions = buildOptions.get(AppleCommandLineOptions.class);
      Optional<DottedVersion> xcodeVersionFlag = getXcodeVersion(env, appleOptions);
      AppleConfiguration configuration = new AppleConfiguration(appleOptions, xcodeVersionFlag);

      validate(configuration);
      return configuration;
    }

    private void validate(AppleConfiguration config)
        throws InvalidConfigurationException {
      Optional<DottedVersion> xcodeVersion = config.getXcodeVersion();
      if (config.getBitcodeMode() != AppleBitcodeMode.NONE
          && xcodeVersion.isPresent()
          && xcodeVersion.get().compareTo(MINIMUM_BITCODE_XCODE_VERSION) < 0) {
        throw new InvalidConfigurationException(
            String.format("apple_bitcode mode '%s' is unsupported for xcode version '%s'",
                config.getBitcodeMode(), xcodeVersion.get()));
      }
    }

    @Override
    public Class<? extends BuildConfiguration.Fragment> creates() {
      return AppleConfiguration.class;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.<Class<? extends FragmentOptions>>of(AppleCommandLineOptions.class);
    }
    
    /**
     * Uses the {@link AppleCommandLineOptions#xcodeVersion} and
     * {@link AppleCommandLineOptions#xcodeVersionConfig} command line options to determine and
     * return the effective xcode version. Returns absent if no explicit xcode version is
     * declared, and host system defaults should be used.
     *
     * @param env the current configuration environment
     * @param appleOptions the command line options
     * @throws InvalidConfigurationException if the options given (or configuration targets) were
     *     malformed and thus the xcode version could not be determined
     */
    private Optional<DottedVersion> getXcodeVersion(ConfigurationEnvironment env,
        AppleCommandLineOptions appleOptions) throws InvalidConfigurationException {
      Optional<DottedVersion> xcodeVersionCommandLineFlag = 
          Optional.fromNullable(appleOptions.xcodeVersion);
      Label xcodeVersionConfigLabel = appleOptions.xcodeVersionConfig;

      return XcodeConfig.resolveXcodeVersion(env, xcodeVersionConfigLabel,
          xcodeVersionCommandLineFlag, "xcode_version_config");
    }
  }
}
