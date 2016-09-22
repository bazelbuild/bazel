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

import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions.AppleBitcodeMode;
import com.google.devtools.build.lib.rules.apple.Platform.PlatformType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import javax.annotation.Nullable;

/** A configuration containing flags required for Apple platforms and tools. */
@SkylarkModule(
  name = "apple",
  doc = "A configuration fragment for Apple platforms",
  category = SkylarkModuleCategory.CONFIGURATION_FRAGMENT
)
@Immutable
public class AppleConfiguration extends BuildConfiguration.Fragment {
  /**
   * Environment variable name for the xcode version. The value of this environment variable should
   * be set to the version (for example, "7.2") of xcode to use when invoking part of the apple
   * toolkit in action execution.
   **/
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
  private final DottedVersion watchOsSdkVersion;
  private final DottedVersion tvOsSdkVersion;
  private final DottedVersion macOsXSdkVersion;
  private final String iosCpu;
  private final String appleSplitCpu;
  private final PlatformType applePlatformType;
  private final ConfigurationDistinguisher configurationDistinguisher;
  private final Optional<DottedVersion> xcodeVersion;
  private final ImmutableList<String> iosMultiCpus;
  private final ImmutableList<String> watchosCpus;
  private final ImmutableList<String> tvosCpus;
  private final AppleBitcodeMode bitcodeMode;
  private final Label xcodeConfigLabel;
  @Nullable private final Label defaultProvisioningProfileLabel;

  AppleConfiguration(AppleCommandLineOptions appleOptions,
      Optional<DottedVersion> xcodeVersionOverride,
      DottedVersion iosSdkVersion,
      DottedVersion watchOsSdkVersion,
      DottedVersion tvOsSdkVersion,
      DottedVersion macOsXSdkVersion) {
    this.iosSdkVersion = Preconditions.checkNotNull(iosSdkVersion, "iosSdkVersion");
    this.watchOsSdkVersion =
        Preconditions.checkNotNull(watchOsSdkVersion, "watchOsSdkVersion");
    this.tvOsSdkVersion =
        Preconditions.checkNotNull(tvOsSdkVersion, "tvOsSdkVersion");
    this.macOsXSdkVersion =
        Preconditions.checkNotNull(macOsXSdkVersion, "macOsXSdkVersion");

    this.xcodeVersion = Preconditions.checkNotNull(xcodeVersionOverride);
    this.iosCpu = Preconditions.checkNotNull(appleOptions.iosCpu, "iosCpu");
    this.appleSplitCpu = Preconditions.checkNotNull(appleOptions.appleSplitCpu, "appleSplitCpu");
    this.applePlatformType =
        Preconditions.checkNotNull(appleOptions.applePlatformType, "applePlatformType");
    this.configurationDistinguisher = appleOptions.configurationDistinguisher;
    this.iosMultiCpus = ImmutableList.copyOf(
        Preconditions.checkNotNull(appleOptions.iosMultiCpus, "iosMultiCpus"));
    this.watchosCpus = (appleOptions.watchosCpus == null || appleOptions.watchosCpus.isEmpty())
        ? ImmutableList.of(AppleCommandLineOptions.DEFAULT_WATCHOS_CPU)
        : ImmutableList.copyOf(appleOptions.watchosCpus);
    this.tvosCpus = (appleOptions.tvosCpus == null || appleOptions.tvosCpus.isEmpty())
        ? ImmutableList.of(AppleCommandLineOptions.DEFAULT_TVOS_CPU)
        : ImmutableList.copyOf(appleOptions.tvosCpus);
    this.bitcodeMode = appleOptions.appleBitcodeMode;
    this.xcodeConfigLabel =
        Preconditions.checkNotNull(appleOptions.xcodeVersionConfig, "xcodeConfigLabel");
    this.defaultProvisioningProfileLabel = appleOptions.defaultProvisioningProfile;
  }

  /**
   * Returns the SDK version for ios SDKs (whether they be for simulator or device). This is
   * directly derived from --ios_sdk_version.
   *
   * @deprecated - use {@link #getSdkVersionForPlatform()}
   */
  @Deprecated public DottedVersion getIosSdkVersion() {
    return getSdkVersionForPlatform(Platform.IOS_DEVICE);
  }

  /**
   * Returns the SDK version for a platform (whether they be for simulator or device). This is
   * directly derived from command line args.
   */
  @SkylarkCallable(name = "sdk_version_for_platform", doc = "The SDK version given a platform.")
  public DottedVersion getSdkVersionForPlatform(Platform platform) {
    switch (platform) {
      case IOS_DEVICE:
      case IOS_SIMULATOR:
        return iosSdkVersion;
      case TVOS_DEVICE:
      case TVOS_SIMULATOR:
        return tvOsSdkVersion;
      case WATCHOS_DEVICE:
      case WATCHOS_SIMULATOR:
        return watchOsSdkVersion;
      case MACOS_X:
        return macOsXSdkVersion;
    }
    throw new AssertionError();

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
   * for actions pertaining to the given apple platform. Keys are variable names and values are
   * their corresponding values.
   */
  @SkylarkCallable(name = "target_apple_env")
  public Map<String, String> getTargetAppleEnvironment(Platform platform) {
    ImmutableMap.Builder<String, String> mapBuilder = ImmutableMap.builder();
    mapBuilder.putAll(appleTargetPlatformEnv(platform));
    return mapBuilder.build();
  }

  /**
   * Returns a map of environment variables that should be propagated for actions that build on an
   * apple host system. These environment variables are needed by the apple toolchain. Keys are
   * variable names and values are their corresponding values.
   */
  @SkylarkCallable(
      name = "apple_host_system_env",
      doc =
          "Returns a map of environment variables that should be propagated for actions that "
          + "build on an apple host system. These environment variables are needed by the apple "
          + "toolchain. Keys are variable names and values are their corresponding values."
    )
  public Map<String, String> getAppleHostSystemEnv() {
    Optional<DottedVersion> xcodeVersion = getXcodeVersion();
    if (xcodeVersion.isPresent()) {
      return getXcodeVersionEnv(xcodeVersion.get());
    } else {
      return ImmutableMap.of();
    }
  }

  /**
   * Returns a map of environment variables that should be propagated for actions that require
   * a version of xcode to be explicitly declared. Keys are variable names and values are their
   * corresponding values.
   */
  public Map<String, String> getXcodeVersionEnv(DottedVersion xcodeVersion) {
    return ImmutableMap.of(AppleConfiguration.XCODE_VERSION_ENV_NAME, xcodeVersion.toString());
  }

  /**
   * Returns a map of environment variables (derived from configuration) that should be propagated
   * for actions pertaining to building applications for apple platforms. These environment
   * variables are needed to use apple toolkits. Keys are variable names and values are their
   * corresponding values.
   */
  public Map<String, String> appleTargetPlatformEnv(Platform platform) {
    ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();

    // TODO(cparsons): Avoid setting SDK version for macosx. Until SDK version is
    // evaluated for the current configuration xcode version, this would break users who build
    // cc_* rules without specifying both xcode_version and macosx_sdk_version build options.
    if (platform != Platform.MACOS_X) {
        String sdkVersion = getSdkVersionForPlatform(platform).toString();
        builder.put(AppleConfiguration.APPLE_SDK_VERSION_ENV_NAME, sdkVersion)
            .put(AppleConfiguration.APPLE_SDK_PLATFORM_ENV_NAME, platform.getNameInPlist());
    }
    return builder.build();
  }

  /**
   * Returns the value of {@code ios_cpu} for this configuration. This is not necessarily the
   * platform or cpu for all actions spawned in this configuration; it is appropriate for
   * identifying the target cpu of iOS compile and link actions within this configuration.
   */
  @SkylarkCallable(name = "ios_cpu", doc = "The value of ios_cpu for this configuration.")
  public String getIosCpu() {
    return iosCpu;
  }

  /**
   * Gets the single "effective" architecture for this configuration's {@link PlatformType} (for
   * example, "i386" or "arm64"). Prefer this over {@link #getMultiArchitectures(PlatformType)} only
   * if in the context of rule logic which is only concerned with a single architecture (such as in
   * {@code objc_library}, which registers single-architecture compile actions).
   *
   * <p>Single effective architecture is determined using the following rules:
   *
   * <ol>
   * <li>If {@code --apple_split_cpu} is set (done via prior configuration transition), then that is
   *     the effective architecture.
   * <li>If the multi cpus flag (e.g. {@code --ios_multi_cpus}) is set and non-empty, then the first
   *     such architecture is returned.
   * <li>In the case of iOS, use {@code --ios_cpu} for backwards compatibility.
   * <li>Use the default.
   * </ol>
   */
  @SkylarkCallable(
    name = "single_arch_cpu",
    structField = true,
    doc =
        "The single \"effective\" architecture for this configuration (e.g. i386 or arm64) "
            + "in the context of rule logic which is only concerned with a single architecture "
            + "(such as in objc_library, which registers single-architecture compile actions). "
  )
  public String getSingleArchitecture() {
    if (!Strings.isNullOrEmpty(appleSplitCpu)) {
      return appleSplitCpu;
    }
    switch (applePlatformType) {
      case IOS:
        if (!getIosMultiCpus().isEmpty()) {
          return getIosMultiCpus().get(0);
        } else {
          return getIosCpu();
        }
      case WATCHOS:
        return watchosCpus.get(0);
      case TVOS:
        return tvosCpus.get(0);
      // TODO(cparsons): Handle all platform types.
      default: 
        throw new IllegalArgumentException("Unhandled platform type " + applePlatformType);
    }
  }
 
  /**
   * Gets the "effective" architecture(s) for the given {@link PlatformType}. For example,
   * "i386" or "arm64". At least one architecture is always returned. Prefer this over
   * {@link #getSingleArchitecture} in rule logic which may support multiple architectures, such
   * as bundling rules.
   * 
   * <p>Effective architecture(s) is determined using the following rules:
   * <ol>
   * <li>If {@code --apple_split_cpu} is set (done via prior configuration transition), then
   * that is the effective architecture.</li>
   * <li>If the multi-cpu flag (for example, {@code --ios_multi_cpus}) is non-empty, then, return
   * all architectures from that flag.</li>
   * <li>In the case of iOS, use {@code --ios_cpu} for backwards compatibility.</li>
   * <li>Use the default.</li></ol>
   * 
   * @throws IllegalArgumentException if {@code --apple_platform_type} is set (via prior
   *     configuration transition) yet does not match {@code platformType}
   */
  public List<String> getMultiArchitectures(PlatformType platformType) {
    if (!Strings.isNullOrEmpty(appleSplitCpu)) {
      if (applePlatformType != platformType) {
        throw new IllegalArgumentException(
            String.format("Expected post-split-transition platform type %s to match input %s ",
                applePlatformType, platformType));
      }
      return ImmutableList.of(appleSplitCpu);
    }
    switch (platformType) {
      case IOS:
        if (getIosMultiCpus().isEmpty()) {
          return ImmutableList.of(getIosCpu());
        } else {
          return getIosMultiCpus();
        }
      case WATCHOS:
        return watchosCpus;
      case TVOS:
        return tvosCpus;
      default: 
        throw new IllegalArgumentException("Unhandled platform type " + platformType);
    }
  }

  /**
   * Gets the single "effective" platform for this configuration's {@link PlatformType} and
   * architecture. Prefer this over {@link #getMultiArchPlatform(PlatformType)} only in cases if in
   * the context of rule logic which is only concerned with a single architecture (such as in {@code
   * objc_library}, which registers single-architecture compile actions).
   */
  @SkylarkCallable(
    name = "single_arch_platform",
    doc =
        "The platform of the current configuration. This should only be invoked in a context where "
            + "only a single architecture may be supported; consider mutli_arch_platform for other "
            + "cases.",
    structField = true
  )
  public Platform getSingleArchPlatform() {
    return Platform.forTarget(applePlatformType, getSingleArchitecture());
  }
  
  /**
   * Gets the current configuration {@link Platform} for the given {@link PlatformType}. Platform
   * is determined via a combination between the given platform type and the "effective"
   * architectures of this configuration, as returned by {@link #getMultiArchitectures}; if any
   * of the supported architectures are of device type, this will return a device platform.
   * Otherwise, this will return a simulator platform.
   */
  // TODO(bazel-team): This should support returning multiple platforms.
  @SkylarkCallable(name = "multi_arch_platform", doc = "The platform of the current configuration "
      + "for the given platform type. This should only be invoked in a context where multiple "
      + "architectures may be supported; consider single_arch_platform for other cases.")
  public Platform getMultiArchPlatform(PlatformType platformType) {
    List<String> architectures = getMultiArchitectures(platformType);
    switch (platformType) {
      case IOS:
        for (String arch : architectures) {
          if (Platform.forTarget(PlatformType.IOS, arch) == Platform.IOS_DEVICE) {
            return Platform.IOS_DEVICE;
          }
        }
        return Platform.IOS_SIMULATOR;
      case WATCHOS:
        for (String arch : architectures) {
          if (Platform.forTarget(PlatformType.WATCHOS, arch) == Platform.WATCHOS_DEVICE) {
            return Platform.WATCHOS_DEVICE;
          }
        }
        return Platform.WATCHOS_SIMULATOR;
      case TVOS:
        for (String arch : architectures) {
          if (Platform.forTarget(PlatformType.TVOS, arch) == Platform.TVOS_DEVICE) {
            return Platform.TVOS_DEVICE;
          }
        }
        return Platform.TVOS_SIMULATOR;
      default:
        throw new IllegalArgumentException("Unsupported platform type " + platformType);
    }
  }

  /**
   * Returns the {@link Platform} represented by {@code ios_cpu} (see {@link #getIosCpu}.
   * (For example, {@code i386} maps to {@link Platform#IOS_SIMULATOR}.) Note that this is not
   * necessarily the effective platform for all ios actions in the current context: This is
   * typically the correct platform for implicityly-ios compile and link actions in the current
   * context. For effective platform for bundling actions, see
   * {@link #getMultiArchPlatform(PlatformType)}.
   */
  // TODO(b/28754442): Deprecate for more general skylark-exposed platform retrieval.
  @SkylarkCallable(name = "ios_cpu_platform", doc = "The platform given by the ios_cpu flag.")
  public Platform getIosCpuPlatform() {
    return Platform.forTarget(PlatformType.IOS, iosCpu);
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
  public ImmutableList<String> getIosMultiCpus() {
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
   * Returns the bitcode mode to use for compilation steps. Users can control bitcode mode using the
   * {@code apple_bitcode} build flag.
   *
   * @see AppleBitcodeMode
   */
  @SkylarkCallable(
    name = "bitcode_mode",
    doc = "Returns the bitcode mode to use for compilation steps.",
    structField = true
  )
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
   * Returns the unique identifier distinguishing configurations that are otherwise the same.
   *
   * <p>Use this value for situations in which two configurations create two outputs that are the
   * same but are not collapsed due to their different configuration owners.
   */
  public ConfigurationDistinguisher getConfigurationDistinguisher() {
    return configurationDistinguisher;
  }


  @Nullable
  @Override
  public String getOutputDirectoryName() {
    List<String> components = new ArrayList<>();
    if (!appleSplitCpu.isEmpty()) {
      components.add(applePlatformType.toString().toLowerCase());
      components.add(appleSplitCpu);
    }
    if (configurationDistinguisher != ConfigurationDistinguisher.UNKNOWN) {
      components.add(configurationDistinguisher.toString().toLowerCase(Locale.US));
    }

    if (components.isEmpty()) {
      return null;
    }
    return Joiner.on('-').join(components);
  }

  /**
   * Loads {@link AppleConfiguration} from build options.
   */
  public static class Loader implements ConfigurationFragmentFactory {
    @Override
    public AppleConfiguration create(ConfigurationEnvironment env, BuildOptions buildOptions)
        throws InvalidConfigurationException, InterruptedException {
      AppleCommandLineOptions appleOptions = buildOptions.get(AppleCommandLineOptions.class);
      XcodeVersionProperties xcodeVersionProperties = getXcodeVersionProperties(env, appleOptions);

      DottedVersion iosSdkVersion = (appleOptions.iosSdkVersion != null)
          ? appleOptions.iosSdkVersion : xcodeVersionProperties.getDefaultIosSdkVersion();
      DottedVersion watchosSdkVersion = (appleOptions.watchOsSdkVersion != null)
          ? appleOptions.watchOsSdkVersion : xcodeVersionProperties.getDefaultWatchosSdkVersion();
      DottedVersion tvosSdkVersion = (appleOptions.tvOsSdkVersion != null)
          ? appleOptions.tvOsSdkVersion : xcodeVersionProperties.getDefaultTvosSdkVersion();
      DottedVersion macosxSdkVersion = (appleOptions.macOsXSdkVersion != null)
          ? appleOptions.macOsXSdkVersion : xcodeVersionProperties.getDefaultMacosxSdkVersion();
      AppleConfiguration configuration =
          new AppleConfiguration(appleOptions, xcodeVersionProperties.getXcodeVersion(),
              iosSdkVersion, watchosSdkVersion, tvosSdkVersion, macosxSdkVersion);

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
     * Uses the {@link AppleCommandLineOptions#xcodeVersion} and {@link
     * AppleCommandLineOptions#xcodeVersionConfig} command line options to determine and return the
     * effective xcode version properties. Returns absent if no explicit xcode version is declared,
     * and host system defaults should be used.
     *
     * @param env the current configuration environment
     * @param appleOptions the command line options
     * @throws InvalidConfigurationException if the options given (or configuration targets) were
     *     malformed and thus the xcode version could not be determined
     */
    private static XcodeVersionProperties getXcodeVersionProperties(
        ConfigurationEnvironment env, AppleCommandLineOptions appleOptions)
        throws InvalidConfigurationException, InterruptedException {
      Optional<DottedVersion> xcodeVersionCommandLineFlag = 
          Optional.fromNullable(appleOptions.xcodeVersion);
      Label xcodeVersionConfigLabel = appleOptions.xcodeVersionConfig;

      return XcodeConfig.resolveXcodeVersion(env, xcodeVersionConfigLabel,
          xcodeVersionCommandLineFlag, "xcode_version_config");
    }
  }

  /**
   * Value used to avoid multiple configurations from conflicting. No two instances of this
   * transition may exist with the same value in a single Bazel invocation.
   */
  public enum ConfigurationDistinguisher {
    UNKNOWN,
    /** Split transition distinguisher for {@code ios_extension} rule. */
    IOS_EXTENSION,
    /** Split transition distinguisher for {@code ios_application} rule. */
    IOS_APPLICATION,
    /** Split transition distinguisher for {@code ios_framework} rule. */
    FRAMEWORK,
    /** Split transition distinguisher for {@code apple_watch1_extension} rule. */
    WATCH_OS1_EXTENSION,
    /** Distinguisher for {@code apple_binary} rule with "ios" platform_type. */
    APPLEBIN_IOS,
    /** Distinguisher for {@code apple_binary} rule with "watchos" platform_type. */
    APPLEBIN_WATCHOS,
    /** Distinguisher for {@code apple_binary} rule with "tvos" platform_type. */
    APPLEBIN_TVOS,
  }
}
