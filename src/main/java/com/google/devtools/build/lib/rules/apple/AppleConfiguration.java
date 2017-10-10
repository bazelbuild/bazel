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

import static com.google.devtools.build.lib.skyframe.serialization.SerializationCommonUtils.deserializeNullable;
import static com.google.devtools.build.lib.skyframe.serialization.SerializationCommonUtils.serializeNullable;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions.AppleBitcodeMode;
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.skyframe.serialization.EnumCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.strings.StringCodecs;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import javax.annotation.Nullable;

/** A configuration containing flags required for Apple platforms and tools. */
@SkylarkModule(
  name = "apple",
  doc = "A configuration fragment for Apple platforms.",
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
   * require an apple SDK. The valid values consist of {@link ApplePlatform} names.
   */
  public static final String APPLE_SDK_PLATFORM_ENV_NAME = "APPLE_SDK_PLATFORM";

  private static final DottedVersion MINIMUM_BITCODE_XCODE_VERSION = DottedVersion.fromString("7");

  /** Prefix for iOS cpu values. */
  public static final String IOS_CPU_PREFIX = "ios_";

  /** Default cpu for iOS builds. */
  @VisibleForTesting static final String DEFAULT_IOS_CPU = "x86_64";

  @Nullable private final DottedVersion xcodeVersion;
  private final DottedVersion iosSdkVersion;
  private final DottedVersion iosMinimumOs;
  private final DottedVersion watchosSdkVersion;
  private final DottedVersion watchosMinimumOs;
  private final DottedVersion tvosSdkVersion;
  private final DottedVersion tvosMinimumOs;
  private final DottedVersion macosSdkVersion;
  private final DottedVersion macosMinimumOs;
  private final String iosCpu;
  private final String appleSplitCpu;
  private final PlatformType applePlatformType;
  private final ConfigurationDistinguisher configurationDistinguisher;
  private final ImmutableList<String> iosMultiCpus;
  private final ImmutableList<String> watchosCpus;
  private final ImmutableList<String> tvosCpus;
  private final ImmutableList<String> macosCpus;
  private final AppleBitcodeMode bitcodeMode;
  private final Label xcodeConfigLabel;
  private final boolean enableAppleCrosstool;
  private final AppleCommandLineOptions options;
  @Nullable private final String xcodeToolchain;
  @Nullable private final Label defaultProvisioningProfileLabel;
  private final boolean mandatoryMinimumVersion;
  private final boolean objcProviderFromLinked;

  @VisibleForTesting
  AppleConfiguration(
      AppleCommandLineOptions options,
      String iosCpu,
      @Nullable DottedVersion xcodeVersion,
      DottedVersion iosSdkVersion,
      DottedVersion iosMinimumOs,
      DottedVersion watchosSdkVersion,
      DottedVersion watchosMinimumOs,
      DottedVersion tvosSdkVersion,
      DottedVersion tvosMinimumOs,
      DottedVersion macosSdkVersion,
      DottedVersion macosMinimumOs) {
    this.options = options;
    this.iosSdkVersion = Preconditions.checkNotNull(iosSdkVersion, "iosSdkVersion");
    this.iosMinimumOs = Preconditions.checkNotNull(iosMinimumOs, "iosMinimumOs");
    this.watchosSdkVersion =
        Preconditions.checkNotNull(watchosSdkVersion, "watchOsSdkVersion");
    this.watchosMinimumOs =
        Preconditions.checkNotNull(watchosMinimumOs, "watchOsMinimumOs");
    this.tvosSdkVersion =
        Preconditions.checkNotNull(tvosSdkVersion, "tvOsSdkVersion");
    this.tvosMinimumOs =
        Preconditions.checkNotNull(tvosMinimumOs, "tvOsMinimumOs");
    this.macosSdkVersion =
        Preconditions.checkNotNull(macosSdkVersion, "macOsSdkVersion");
    this.macosMinimumOs = Preconditions.checkNotNull(macosMinimumOs, "macOsMinimumOs");

    this.xcodeVersion = xcodeVersion;
    this.iosCpu = iosCpu;
    this.appleSplitCpu = Preconditions.checkNotNull(options.appleSplitCpu, "appleSplitCpu");
    this.applePlatformType =
        Preconditions.checkNotNull(options.applePlatformType, "applePlatformType");
    this.configurationDistinguisher = options.configurationDistinguisher;
    this.iosMultiCpus = ImmutableList.copyOf(
        Preconditions.checkNotNull(options.iosMultiCpus, "iosMultiCpus"));
    this.watchosCpus = (options.watchosCpus == null || options.watchosCpus.isEmpty())
        ? ImmutableList.of(AppleCommandLineOptions.DEFAULT_WATCHOS_CPU)
        : ImmutableList.copyOf(options.watchosCpus);
    this.tvosCpus = (options.tvosCpus == null || options.tvosCpus.isEmpty())
        ? ImmutableList.of(AppleCommandLineOptions.DEFAULT_TVOS_CPU)
        : ImmutableList.copyOf(options.tvosCpus);
    this.macosCpus = (options.macosCpus == null || options.macosCpus.isEmpty())
        ? ImmutableList.of(AppleCommandLineOptions.DEFAULT_MACOS_CPU)
        : ImmutableList.copyOf(options.macosCpus);
    this.bitcodeMode = options.appleBitcodeMode;
    this.xcodeConfigLabel =
        Preconditions.checkNotNull(options.xcodeVersionConfig, "xcodeConfigLabel");
    this.enableAppleCrosstool = options.enableAppleCrosstoolTransition;
    this.defaultProvisioningProfileLabel = options.defaultProvisioningProfile;
    this.xcodeToolchain = options.xcodeToolchain;
    this.mandatoryMinimumVersion = options.mandatoryMinimumVersion;
    this.objcProviderFromLinked = options.objcProviderFromLinked;
  }

  /** Determines cpu value from apple-specific toolchain identifier. */
  public static String iosCpuFromCpu(String cpu) {
    if (cpu.startsWith(IOS_CPU_PREFIX)) {
      return cpu.substring(IOS_CPU_PREFIX.length());
    } else {
      return DEFAULT_IOS_CPU;
    }
  }

  public AppleCommandLineOptions getOptions() {
    return options;
  }

  /***
   * @deprecated use {@link XcodeConfig#getMinimumOsForPlatformType(RuleContext, PlatformType)}.
   *
   * <p>This is only here because the minimum OS version is currently part of the name of the output
   * directory.
   */
  @Deprecated
  // Bug tracking the removal of this method: https://github.com/bazelbuild/bazel/issues/3424
  public DottedVersion getMinimumOsForPlatformType(PlatformType platformType) {
    // TODO(b/37240784): Look into using only a single minimum OS flag tied to the current
    // apple_platform_type.
    switch (platformType) {
      case IOS:
        return iosMinimumOs;
      case TVOS:
        return tvosMinimumOs;
      case WATCHOS:
        return watchosMinimumOs;
      case MACOS:
        return macosMinimumOs;
      default:
        throw new IllegalArgumentException("Unhandled platform: " + platformType);
    }
  }


  /**
   * Returns the SDK version for a platform (whether they be for simulator or device). This is
   * directly derived from command line args.
   *
   * @deprecated use {@link XcodeConfig#getSdkVersionForPlatform(RuleContext, ApplePlatform)}
   */
  @Deprecated
  // Bug tracking the removal of this method: https://github.com/bazelbuild/bazel/issues/3424
  private DottedVersion getSdkVersionForPlatform(ApplePlatform platform) {
    switch (platform) {
      case IOS_DEVICE:
      case IOS_SIMULATOR:
        return iosSdkVersion;
      case TVOS_DEVICE:
      case TVOS_SIMULATOR:
        return tvosSdkVersion;
      case WATCHOS_DEVICE:
      case WATCHOS_SIMULATOR:
        return watchosSdkVersion;
      case MACOS:
        return macosSdkVersion;
    }
    throw new AssertionError();

  }

  /**
   * Returns a map of environment variables (derived from configuration) that should be propagated
   * for actions pertaining to the given apple platform. Keys are variable names and values are
   * their corresponding values.
   */
  @SkylarkCallable(
      name = "target_apple_env",
      doc = "Returns a <code>dict</code> of environment variables that should be set for actions "
          + "that build targets of the given Apple platform type. For example, this dictionary "
          + "contains variables that denote the platform name and SDK version with which to "
          + "build. The keys are variable names and the values are their corresponding values.")
  public ImmutableMap<String, String> getTargetAppleEnvironment(ApplePlatform platform) {
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
      doc = "Returns a <a href='dict.html'>dict</a> of environment variables that should be set "
          + "for actions that need to run build tools on an Apple host system, such as the version "
          + "of Xcode that should be used. The keys are variable names and the values are their "
          + "corresponding values.")
  public ImmutableMap<String, String> getAppleHostSystemEnv() {
    if (xcodeVersion != null) {
      return getXcodeVersionEnv(xcodeVersion);
    } else {
      return ImmutableMap.of();
    }
  }

  /**
   * Returns a map of environment variables that should be propagated for actions that require
   * a version of xcode to be explicitly declared. Keys are variable names and values are their
   * corresponding values.
   */
  public ImmutableMap<String, String> getXcodeVersionEnv(DottedVersion xcodeVersion) {
    return ImmutableMap.of(AppleConfiguration.XCODE_VERSION_ENV_NAME, xcodeVersion.toString());
  }

  /**
   * Returns a map of environment variables (derived from configuration) that should be propagated
   * for actions pertaining to building applications for apple platforms. These environment
   * variables are needed to use apple toolkits. Keys are variable names and values are their
   * corresponding values.
   */
  public Map<String, String> appleTargetPlatformEnv(ApplePlatform platform) {
    ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();

    String sdkVersion = getSdkVersionForPlatform(platform).toStringWithMinimumComponents(2);
    builder
        .put(AppleConfiguration.APPLE_SDK_VERSION_ENV_NAME, sdkVersion)
        .put(AppleConfiguration.APPLE_SDK_PLATFORM_ENV_NAME, platform.getNameInPlist());

    return builder.build();
  }

  /**
   * Returns the value of {@code ios_cpu} for this configuration. This is not necessarily the
   * platform or cpu for all actions spawned in this configuration; it is appropriate for
   * identifying the target cpu of iOS compile and link actions within this configuration.
   */
  @SkylarkCallable(
      name = "ios_cpu",
      doc = "<b>Deprecated. Use <a href='#single_arch_cpu'>single_arch_cpu</a> instead.</b> "
          + "The value of ios_cpu for this configuration.")
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
        "The single \"effective\" architecture for this configuration (e.g., <code>i386</code> or "
            + "<code>arm64</code>) in the context of rule logic that is only concerned with a "
            + "single architecture (such as <code>objc_library</code>, which registers "
            + "single-architecture compile actions)."
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
      case MACOS:
        return macosCpus.get(0);
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
      case MACOS:
        return macosCpus;
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
    doc = "The platform of the current configuration. This should only be invoked in a context "
        + "where only a single architecture may be supported; consider "
        + "<a href='#multi_arch_platform'>multi_arch_platform</a> for other cases.",
    structField = true
  )
  public ApplePlatform getSingleArchPlatform() {
    return ApplePlatform.forTarget(applePlatformType, getSingleArchitecture());
  }

  private boolean hasValidSingleArchPlatform() {
    return ApplePlatform.isApplePlatform(
        ApplePlatform.cpuStringForTarget(applePlatformType, getSingleArchitecture()));
  }

  /**
   * Gets the current configuration {@link ApplePlatform} for the given {@link PlatformType}.
   * ApplePlatform is determined via a combination between the given platform type and the
   * "effective" architectures of this configuration, as returned by {@link #getMultiArchitectures};
   * if any of the supported architectures are of device type, this will return a device platform.
   * Otherwise, this will return a simulator platform.
   */
  // TODO(bazel-team): This should support returning multiple platforms.
  @SkylarkCallable(
    name = "multi_arch_platform",
    doc = "The platform of the current configuration for the given platform type. This should only "
        + "be invoked in a context where multiple architectures may be supported; consider "
        + "<a href='#single_arch_platform'>single_arch_platform</a> for other cases."
  )
  public ApplePlatform getMultiArchPlatform(PlatformType platformType) {
    List<String> architectures = getMultiArchitectures(platformType);
    switch (platformType) {
      case IOS:
        for (String arch : architectures) {
          if (ApplePlatform.forTarget(PlatformType.IOS, arch) == ApplePlatform.IOS_DEVICE) {
            return ApplePlatform.IOS_DEVICE;
          }
        }
        return ApplePlatform.IOS_SIMULATOR;
      case WATCHOS:
        for (String arch : architectures) {
          if (ApplePlatform.forTarget(PlatformType.WATCHOS, arch) == ApplePlatform.WATCHOS_DEVICE) {
            return ApplePlatform.WATCHOS_DEVICE;
          }
        }
        return ApplePlatform.WATCHOS_SIMULATOR;
      case TVOS:
        for (String arch : architectures) {
          if (ApplePlatform.forTarget(PlatformType.TVOS, arch) == ApplePlatform.TVOS_DEVICE) {
            return ApplePlatform.TVOS_DEVICE;
          }
        }
        return ApplePlatform.TVOS_SIMULATOR;
      case MACOS:
        return ApplePlatform.MACOS;
      default:
        throw new IllegalArgumentException("Unsupported platform type " + platformType);
    }
  }

  /**
   * Returns the {@link ApplePlatform} represented by {@code ios_cpu} (see {@link #getIosCpu}. (For
   * example, {@code i386} maps to {@link ApplePlatform#IOS_SIMULATOR}.) Note that this is not
   * necessarily the effective platform for all ios actions in the current context: This is
   * typically the correct platform for implicityly-ios compile and link actions in the current
   * context. For effective platform for bundling actions, see {@link
   * #getMultiArchPlatform(PlatformType)}.
   */
  // TODO(b/28754442): Deprecate for more general skylark-exposed platform retrieval.
  @SkylarkCallable(
      name = "ios_cpu_platform",
      doc = "<b>Deprecated. Use <a href='#single_arch_platform'>single_arch_platform</a> or "
          + "<a href='#multi_arch_platform'>multi_arch_platform</a> instead.</b> "
          + "The platform given by the ios_cpu flag.")
  public ApplePlatform getIosCpuPlatform() {
    return ApplePlatform.forTarget(PlatformType.IOS, iosCpu);
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
   * Returns the bitcode mode to use for compilation steps. This should only be invoked in
   * single-architecture contexts.
   *
   * <p>Users can control bitcode mode using the {@code apple_bitcode} build flag, but bitcode
   * will be disabled for all simulator architectures regardless of this flag.
   *
   * @see AppleBitcodeMode
   */
  @SkylarkCallable(
    name = "bitcode_mode",
    doc = "Returns the Bitcode mode to use for compilation steps.<p>"
        + "This field is only valid for device builds; for simulator builds, it always returns "
        + "<code>'none'</code>.",
    structField = true
  )
  public AppleBitcodeMode getBitcodeMode() {
    if (hasValidSingleArchPlatform() && getSingleArchPlatform().isDevice()) {
      return bitcodeMode;
    } else {
      return AppleBitcodeMode.NONE;
    }
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

  private boolean shouldDistinguishOutputDirectory() {
    if (configurationDistinguisher == ConfigurationDistinguisher.UNKNOWN) {
      return false;
    } else if (configurationDistinguisher == ConfigurationDistinguisher.APPLE_CROSSTOOL
        && isAppleCrosstoolEnabled()) {
      return false;
    } else {
      return true;
    }
  }

  @Nullable
  @Override
  public String getOutputDirectoryName() {
    List<String> components = new ArrayList<>();
    if (!appleSplitCpu.isEmpty()) {
      components.add(applePlatformType.toString().toLowerCase());
      components.add(appleSplitCpu);

      if (!isMandatoryMinimumVersion() || options.isMinimumOsVersionExplicitlySet()) {
        components.add("min" + getMinimumOsForPlatformType(applePlatformType));
      }
    }
    if (shouldDistinguishOutputDirectory()) {
      components.add(configurationDistinguisher.getFileSystemName());
    }

    if (components.isEmpty()) {
      return null;
    }
    return Joiner.on('-').join(components);
  }

  /** Returns the identifier for an Xcode toolchain to use with tools. */
  @SkylarkCallable(
    name = "xcode_toolchain",
    doc = "Identifier for the custom Xcode toolchain to use in build, or <code>None</code> if it "
        + "is not specified.",
    allowReturnNones = true,
    structField = true
  )
  public String getXcodeToolchain() {
    return xcodeToolchain;
  }

  /** Returns true if the minimum_os_version attribute should be mandatory on rules with linking. */
  public boolean isMandatoryMinimumVersion() {
    return mandatoryMinimumVersion;
  }

  /**
   * Returns true if rules which manage link actions should propagate {@link ObjcProvider} at the
   * top level.
   **/
  public boolean shouldLinkingRulesPropagateObjc() {
    return objcProviderFromLinked;
  }

  /** Returns true if {@link AppleCrosstoolTransition} should be applied to every apple rule. */
  public boolean isAppleCrosstoolEnabled() {
    return enableAppleCrosstool;
  }

  @Override
  public Map<String, Object> lateBoundOptionDefaults() {
    // xcode_version and *_sdk_version defaults come from processing the
    // target with label given in --xcode_version_override.
    ImmutableMap.Builder<String, Object> mapBuilder = ImmutableMap.builder();

    if (xcodeVersion != null) {
      mapBuilder.put("xcode_version", xcodeVersion.toString());
    }
    return mapBuilder
        .put("ios_sdk_version", iosSdkVersion)
        .put("tvos_sdk_version", tvosSdkVersion)
        .put("watchos_sdk_version", watchosSdkVersion)
        .put("macos_sdk_version", macosSdkVersion)
        .build();
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof AppleConfiguration)) {
      return false;
    }
    AppleConfiguration that = (AppleConfiguration) obj;
    return this.options.equals(that.options)
        && Objects.equals(this.xcodeVersion, that.xcodeVersion)
        && this.iosSdkVersion.equals(that.iosSdkVersion)
        && this.iosMinimumOs.equals(that.iosMinimumOs)
        && this.watchosSdkVersion.equals(that.watchosSdkVersion)
        && this.watchosMinimumOs.equals(that.watchosMinimumOs)
        && this.tvosSdkVersion.equals(that.tvosSdkVersion)
        && this.tvosMinimumOs.equals(that.tvosMinimumOs)
        && this.macosSdkVersion.equals(that.macosSdkVersion)
        && this.macosMinimumOs.equals(that.macosMinimumOs);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        options,
        xcodeVersion,
        iosSdkVersion,
        iosMinimumOs,
        watchosSdkVersion,
        watchosMinimumOs,
        tvosSdkVersion,
        tvosMinimumOs,
        macosSdkVersion,
        macosMinimumOs);
  }

  void serialize(CodedOutputStream out) throws IOException, SerializationException {
    options.serialize(out);
    out.writeStringNoTag(iosCpu);
    serializeNullable(xcodeVersion, out, DottedVersion.CODEC);
    DottedVersion.CODEC.serialize(iosSdkVersion, out);
    DottedVersion.CODEC.serialize(iosMinimumOs, out);
    DottedVersion.CODEC.serialize(watchosSdkVersion, out);
    DottedVersion.CODEC.serialize(watchosMinimumOs, out);
    DottedVersion.CODEC.serialize(tvosSdkVersion, out);
    DottedVersion.CODEC.serialize(tvosMinimumOs, out);
    DottedVersion.CODEC.serialize(macosSdkVersion, out);
    DottedVersion.CODEC.serialize(macosMinimumOs, out);
  }

  static AppleConfiguration deserialize(CodedInputStream in)
      throws IOException, SerializationException {
    AppleCommandLineOptions options = AppleCommandLineOptions.deserialize(in);
    String iosCpu = StringCodecs.asciiOptimized().deserialize(in);
    DottedVersion xcodeVersion = deserializeNullable(in, DottedVersion.CODEC);
    return new AppleConfiguration(
        options,
        iosCpu,
        xcodeVersion,
        DottedVersion.CODEC.deserialize(in),
        DottedVersion.CODEC.deserialize(in),
        DottedVersion.CODEC.deserialize(in),
        DottedVersion.CODEC.deserialize(in),
        DottedVersion.CODEC.deserialize(in),
        DottedVersion.CODEC.deserialize(in),
        DottedVersion.CODEC.deserialize(in),
        DottedVersion.CODEC.deserialize(in));
  }

  @VisibleForTesting
  static AppleConfiguration create(
      AppleCommandLineOptions appleOptions,
      String cpu,
      XcodeVersionProperties xcodeVersionProperties)
      throws InvalidConfigurationException {
    DottedVersion iosSdkVersion =
        (appleOptions.iosSdkVersion != null)
            ? appleOptions.iosSdkVersion
            : xcodeVersionProperties.getDefaultIosSdkVersion();
    DottedVersion iosMinimumOsVersion =
        (appleOptions.iosMinimumOs != null) ? appleOptions.iosMinimumOs : iosSdkVersion;
    DottedVersion watchosSdkVersion =
        (appleOptions.watchOsSdkVersion != null)
            ? appleOptions.watchOsSdkVersion
            : xcodeVersionProperties.getDefaultWatchosSdkVersion();
    DottedVersion watchosMinimumOsVersion =
        (appleOptions.watchosMinimumOs != null) ? appleOptions.watchosMinimumOs : watchosSdkVersion;
    DottedVersion tvosSdkVersion =
        (appleOptions.tvOsSdkVersion != null)
            ? appleOptions.tvOsSdkVersion
            : xcodeVersionProperties.getDefaultTvosSdkVersion();
    DottedVersion tvosMinimumOsVersion =
        (appleOptions.tvosMinimumOs != null) ? appleOptions.tvosMinimumOs : tvosSdkVersion;
    DottedVersion macosSdkVersion =
        (appleOptions.macOsSdkVersion != null)
            ? appleOptions.macOsSdkVersion
            : xcodeVersionProperties.getDefaultMacosSdkVersion();
    DottedVersion macosMinimumOsVersion =
        (appleOptions.macosMinimumOs != null) ? appleOptions.macosMinimumOs : macosSdkVersion;
    AppleConfiguration configuration =
        new AppleConfiguration(
            appleOptions,
            iosCpuFromCpu(cpu),
            xcodeVersionProperties.getXcodeVersion().orNull(),
            iosSdkVersion,
            iosMinimumOsVersion,
            watchosSdkVersion,
            watchosMinimumOsVersion,
            tvosSdkVersion,
            tvosMinimumOsVersion,
            macosSdkVersion,
            macosMinimumOsVersion);

    validate(configuration);
    return configuration;
  }

  private static void validate(AppleConfiguration config) throws InvalidConfigurationException {
    DottedVersion xcodeVersion = config.xcodeVersion;
    if (config.getBitcodeMode() != AppleBitcodeMode.NONE
        && xcodeVersion != null
        && xcodeVersion.compareTo(MINIMUM_BITCODE_XCODE_VERSION) < 0) {
      throw new InvalidConfigurationException(
          String.format(
              "apple_bitcode mode '%s' is unsupported for xcode version '%s'",
              config.getBitcodeMode(), xcodeVersion));
    }
  }

  /**
   * Loads {@link AppleConfiguration} from build options.
   */
  public static class Loader implements ConfigurationFragmentFactory {
    @Override
    public AppleConfiguration create(ConfigurationEnvironment env, BuildOptions buildOptions)
        throws InvalidConfigurationException, InterruptedException {
      AppleCommandLineOptions appleOptions = buildOptions.get(AppleCommandLineOptions.class);
      String cpu = buildOptions.get(BuildConfiguration.Options.class).cpu;
      XcodeVersionProperties xcodeVersionProperties = XcodeConfig.
          getXcodeVersionProperties(env, appleOptions);
      return AppleConfiguration.create(appleOptions, cpu, xcodeVersionProperties);
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

  /**
   * Value used to avoid multiple configurations from conflicting. No two instances of this
   * transition may exist with the same value in a single Bazel invocation.
   */
  public enum ConfigurationDistinguisher {
    UNKNOWN("unknown"),
    /** Split transition distinguisher for {@code ios_application} rule. */
    IOS_APPLICATION("ios_application"),
    /** Distinguisher for {@code apple_binary} rule with "ios" platform_type. */
    APPLEBIN_IOS("applebin_ios"),
    /** Distinguisher for {@code apple_binary} rule with "watchos" platform_type. */
    APPLEBIN_WATCHOS("applebin_watchos"),
    /** Distinguisher for {@code apple_binary} rule with "tvos" platform_type. */
    APPLEBIN_TVOS("applebin_tvos"),
    /** Distinguisher for {@code apple_binary} rule with "macos" platform_type. */
    APPLEBIN_MACOS("applebin_macos"),

    /**
     * Distinguisher for the apple crosstool configuration.  We use "apl" for output directory
     * names instead of "apple_crosstool" to avoid oversized path names, which can be problematic
     * on OSX.
     */
    APPLE_CROSSTOOL("apl");

    private final String fileSystemName;

    private ConfigurationDistinguisher(String fileSystemName) {
      this.fileSystemName = fileSystemName;
    }

    /**
     * Returns the distinct string that should be used in creating output directories for a
     * configuration with this distinguisher.
     */
    public String getFileSystemName() {
      return fileSystemName;
    }

    static final EnumCodec<ConfigurationDistinguisher> CODEC =
        new EnumCodec<>(ConfigurationDistinguisher.class);
  }
}
