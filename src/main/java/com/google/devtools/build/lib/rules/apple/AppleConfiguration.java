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

import static com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions.DEFAULT_MACOS_CPU;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;
import com.google.devtools.build.lib.analysis.starlark.annotations.StarlarkConfigurationField;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.starlarkbuildapi.apple.AppleConfigurationApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.util.CPU;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.eval.Tuple;

// LINT.IfChange

/** A configuration containing flags required for Apple platforms and tools. */
@Immutable
@RequiresOptions(options = {AppleCommandLineOptions.class})
public class AppleConfiguration extends Fragment implements AppleConfigurationApi {
  /** Environment variable name for the developer dir of the selected Xcode. */
  public static final String DEVELOPER_DIR_ENV_NAME = "DEVELOPER_DIR";

  /**
   * Environment variable name for the Xcode version. The value of this environment variable should
   * be set to the version (for example, "7.2") of Xcode to use when invoking part of the apple
   * toolkit in action execution.
   */
  public static final String XCODE_VERSION_ENV_NAME = "XCODE_VERSION_OVERRIDE";

  /**
   * Environment variable name for the apple SDK platform. This should be set for all actions that
   * require an apple SDK. The valid values consist of {@link ApplePlatform} names.
   */
  public static final String APPLE_SDK_PLATFORM_ENV_NAME = "APPLE_SDK_PLATFORM";

  /** Prefix for simulator environment cpu values */
  public static final String SIMULATOR_ENVIRONMENT_CPU_PREFIX = "sim_";

  /** Prefix for device environment cpu values */
  public static final String DEVICE_ENVIRONMENT_CPU_PREFIX = "device_";

  /** Default cpu for iOS builds. */
  @VisibleForTesting
  static final String DEFAULT_IOS_CPU = CPU.getCurrent() == CPU.AARCH64 ? "sim_arm64" : "x86_64";

  private final String applePlatformType;
  private final ConfigurationDistinguisher configurationDistinguisher;
  private final Label xcodeConfigLabel;
  private final AppleCommandLineOptions options;
  private final AppleCpus appleCpus;
  private final String xcodeVersionFlag;
  private final DottedVersion iosSdkVersionFlag;
  private final DottedVersion macOsSdkVersionFlag;
  private final DottedVersion tvOsSdkVersionFlag;
  private final DottedVersion watchOsSdkVersionFlag;
  private final DottedVersion iosMinimumOsFlag;
  private final DottedVersion macosMinimumOsFlag;
  private final DottedVersion tvosMinimumOsFlag;
  private final DottedVersion watchosMinimumOsFlag;
  private final boolean preferMutualXcode;
  private final boolean includeXcodeExecRequirements;

  public AppleConfiguration(BuildOptions buildOptions) {
    AppleCommandLineOptions options = buildOptions.get(AppleCommandLineOptions.class);
    this.options = options;
    this.appleCpus = AppleCpus.create(options);
    this.applePlatformType =
        Preconditions.checkNotNull(options.applePlatformType, "applePlatformType");
    this.configurationDistinguisher = options.configurationDistinguisher;
    this.xcodeConfigLabel =
        Preconditions.checkNotNull(options.xcodeVersionConfig, "xcodeConfigLabel");
    // AppleConfiguration should not have this knowledge. This is a temporary workaround
    // for Starlarkification, until apple rules are toolchainized.
    this.xcodeVersionFlag = options.xcodeVersion;
    this.iosSdkVersionFlag = DottedVersion.maybeUnwrap(options.iosSdkVersion);
    this.macOsSdkVersionFlag = DottedVersion.maybeUnwrap(options.macOsSdkVersion);
    this.tvOsSdkVersionFlag = DottedVersion.maybeUnwrap(options.tvOsSdkVersion);
    this.watchOsSdkVersionFlag = DottedVersion.maybeUnwrap(options.watchOsSdkVersion);
    this.iosMinimumOsFlag = DottedVersion.maybeUnwrap(options.iosMinimumOs);
    this.macosMinimumOsFlag = DottedVersion.maybeUnwrap(options.macosMinimumOs);
    this.tvosMinimumOsFlag = DottedVersion.maybeUnwrap(options.tvosMinimumOs);
    this.watchosMinimumOsFlag = DottedVersion.maybeUnwrap(options.watchosMinimumOs);
    this.preferMutualXcode = options.preferMutualXcode;
    this.includeXcodeExecRequirements = options.includeXcodeExecutionRequirements;
  }

  /** A class that contains information pertaining to Apple CPUs. */
  @AutoValue
  public abstract static class AppleCpus {
    public static AppleCpus create(AppleCommandLineOptions options) {
      String appleSplitCpu = Preconditions.checkNotNull(options.appleSplitCpu, "appleSplitCpu");
      ImmutableList<String> iosMultiCpus =
          (options.iosMultiCpus == null || options.iosMultiCpus.isEmpty())
              ? ImmutableList.of(DEFAULT_IOS_CPU)
              : ImmutableList.copyOf(options.iosMultiCpus);
      ImmutableList<String> visionosCpus =
          (options.visionosCpus == null || options.visionosCpus.isEmpty())
              ? ImmutableList.of(AppleCommandLineOptions.DEFAULT_VISIONOS_CPU)
              : ImmutableList.copyOf(options.visionosCpus);
      ImmutableList<String> watchosCpus =
          (options.watchosCpus == null || options.watchosCpus.isEmpty())
              ? ImmutableList.of(AppleCommandLineOptions.DEFAULT_WATCHOS_CPU)
              : ImmutableList.copyOf(options.watchosCpus);
      ImmutableList<String> tvosCpus =
          (options.tvosCpus == null || options.tvosCpus.isEmpty())
              ? ImmutableList.of(AppleCommandLineOptions.DEFAULT_TVOS_CPU)
              : ImmutableList.copyOf(options.tvosCpus);
      ImmutableList<String> macosCpus =
          (options.macosCpus == null || options.macosCpus.isEmpty())
              ? ImmutableList.of(DEFAULT_MACOS_CPU)
              : ImmutableList.copyOf(options.macosCpus);
      ImmutableList<String> catalystCpus =
          (options.catalystCpus == null || options.catalystCpus.isEmpty())
              ? ImmutableList.of(AppleCommandLineOptions.DEFAULT_CATALYST_CPU)
              : ImmutableList.copyOf(options.catalystCpus);

      return new AutoValue_AppleConfiguration_AppleCpus(
          appleSplitCpu,
          iosMultiCpus,
          visionosCpus,
          watchosCpus,
          tvosCpus,
          macosCpus,
          catalystCpus);
    }

    abstract String appleSplitCpu();

    abstract ImmutableList<String> iosMultiCpus();

    abstract ImmutableList<String> visionosCpus();

    abstract ImmutableList<String> watchosCpus();

    abstract ImmutableList<String> tvosCpus();

    abstract ImmutableList<String> macosCpus();

    abstract ImmutableList<String> catalystCpus();
  }

  @Override
  public StructApi getAppleCpusForStarlark() throws EvalException {
    Map<String, Object> fields = new HashMap<>();
    fields.put("apple_split_cpu", appleCpus.appleSplitCpu());
    fields.put("ios_multi_cpus", Tuple.copyOf(appleCpus.iosMultiCpus()));
    fields.put("visionos_cpus", Tuple.copyOf(appleCpus.visionosCpus()));
    fields.put("watchos_cpus", Tuple.copyOf(appleCpus.watchosCpus()));
    fields.put("tvos_cpus", Tuple.copyOf(appleCpus.tvosCpus()));
    fields.put("macos_cpus", Tuple.copyOf(appleCpus.macosCpus()));
    fields.put("catalyst_cpus", Tuple.copyOf(appleCpus.catalystCpus()));
    return StructProvider.STRUCT.create(fields, "");
  }

  @Override
  public String getApplePlatformType() {
    return applePlatformType;
  }

  public AppleCommandLineOptions getOptions() {
    return options;
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
   *   <li>If {@code --apple_split_cpu} is set (done via prior configuration transition), then that
   *       is the effective architecture.
   *   <li>If the multi cpus flag (e.g. {@code --ios_multi_cpus}) is set and non-empty, then the
   *       first such architecture is returned.
   *   <li>In the case of iOS, use {@code --cpu} if it leads with "ios_" for backwards
   *       compatibility.
   *   <li>In the case of macOS, use {@code --cpu} if it leads with "darwin_" for backwards
   *       compatibility.
   *   <li>Use the default.
   * </ol>
   */
  @Override
  public String getSingleArchitecture() {
    return getUnprefixedAppleCpu(applePlatformType, appleCpus);
  }

  private static String getUnprefixedAppleCpu(String applePlatformType, AppleCpus appleCpus) {
    // The environment data prefix is removed from the CPU string,
    // - e.g. whether the target CPU is for simulator, device or catalyst.
    //  For older CPUs no environment may be provided.
    String cpu = getPrefixedAppleCpu(applePlatformType, appleCpus);
    if (cpu.startsWith(SIMULATOR_ENVIRONMENT_CPU_PREFIX)) {
      cpu = cpu.substring(SIMULATOR_ENVIRONMENT_CPU_PREFIX.length());
    } else if (cpu.startsWith(DEVICE_ENVIRONMENT_CPU_PREFIX)) {
      cpu = cpu.substring(DEVICE_ENVIRONMENT_CPU_PREFIX.length());
    }
    return cpu;
  }

  private static String getPrefixedAppleCpu(String applePlatformType, AppleCpus appleCpus) {
    if (!Strings.isNullOrEmpty(appleCpus.appleSplitCpu())) {
      return appleCpus.appleSplitCpu();
    }
    switch (applePlatformType) {
      case PlatformType.IOS:
        return appleCpus.iosMultiCpus().get(0);
      case PlatformType.VISIONOS:
        return appleCpus.visionosCpus().get(0);
      case PlatformType.WATCHOS:
        return appleCpus.watchosCpus().get(0);
      case PlatformType.TVOS:
        return appleCpus.tvosCpus().get(0);
      case PlatformType.MACOS:
        return appleCpus.macosCpus().get(0);
      case PlatformType.CATALYST:
        return appleCpus.catalystCpus().get(0);
      default:
        throw new IllegalArgumentException("Unhandled platform type " + applePlatformType);
    }
  }

  /**
   * Gets the "effective" architecture(s) for the given {@link PlatformType}. For example, "i386" or
   * "arm64". At least one architecture is always returned. Prefer this over {@link
   * #getSingleArchitecture} in rule logic which may support multiple architectures, such as
   * bundling rules.
   *
   * <p>Effective architecture(s) is determined using the following rules:
   *
   * <ol>
   *   <li>If {@code --apple_split_cpu} is set (done via prior configuration transition), then that
   *       is the effective architecture.
   *   <li>If the multi cpus flag (e.g. {@code --ios_multi_cpus}) is set and non-empty, return all
   *       architectures from that flag.
   *   <li>In the case of iOS, use {@code --cpu} if it leads with "ios_" for backwards
   *       compatibility.
   *   <li>In the case of macOS, use {@code --cpu} if it leads with "darwin_" for backwards
   *       compatibility.
   *   <li>Use the default.
   * </ol>
   *
   * @throws IllegalArgumentException if {@code --apple_platform_type} is set (via prior
   *     configuration transition) yet does not match {@code platformType}
   */
  public List<String> getMultiArchitectures(String platformType) {
    if (!Strings.isNullOrEmpty(appleCpus.appleSplitCpu())) {
      if (!applePlatformType.equals(platformType)) {
        throw new IllegalArgumentException(
            String.format(
                "Expected post-split-transition platform type %s to match input %s ",
                applePlatformType, platformType));
      }
      return ImmutableList.of(appleCpus.appleSplitCpu());
    }
    switch (platformType) {
      case PlatformType.IOS:
        return appleCpus.iosMultiCpus();
      case PlatformType.VISIONOS:
        return appleCpus.visionosCpus();
      case PlatformType.WATCHOS:
        return appleCpus.watchosCpus();
      case PlatformType.TVOS:
        return appleCpus.tvosCpus();
      case PlatformType.MACOS:
        return appleCpus.macosCpus();
      case PlatformType.CATALYST:
        return appleCpus.catalystCpus();
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
  @Override
  public ApplePlatform getSingleArchPlatform() {
    return ApplePlatform.forTarget(
        applePlatformType, getPrefixedAppleCpu(applePlatformType, appleCpus));
  }

  /**
   * Gets the current configuration {@link ApplePlatform} for the given {@link PlatformType}.
   * ApplePlatform is determined via a combination between the given platform type and the
   * "effective" architectures of this configuration, as returned by {@link #getMultiArchitectures};
   * if any of the supported architectures are of device type, this will return a device platform.
   * Otherwise, this will return a simulator platform.
   */
  // TODO(bazel-team): This should support returning multiple platforms.
  @Override
  public ApplePlatform getMultiArchPlatform(String platformType) {
    List<String> architectures = getMultiArchitectures(platformType);
    switch (platformType) {
      case PlatformType.IOS:
        for (String arch : architectures) {
          if (ApplePlatform.forTarget(PlatformType.IOS, arch) == ApplePlatform.IOS_DEVICE) {
            return ApplePlatform.IOS_DEVICE;
          }
        }
        return ApplePlatform.IOS_SIMULATOR;
      case PlatformType.VISIONOS:
        for (String arch : architectures) {
          if (ApplePlatform.forTarget(PlatformType.VISIONOS, arch)
              == ApplePlatform.VISIONOS_DEVICE) {
            return ApplePlatform.VISIONOS_DEVICE;
          }
        }
        return ApplePlatform.VISIONOS_SIMULATOR;
      case PlatformType.WATCHOS:
        for (String arch : architectures) {
          if (ApplePlatform.forTarget(PlatformType.WATCHOS, arch) == ApplePlatform.WATCHOS_DEVICE) {
            return ApplePlatform.WATCHOS_DEVICE;
          }
        }
        return ApplePlatform.WATCHOS_SIMULATOR;
      case PlatformType.TVOS:
        for (String arch : architectures) {
          if (ApplePlatform.forTarget(PlatformType.TVOS, arch) == ApplePlatform.TVOS_DEVICE) {
            return ApplePlatform.TVOS_DEVICE;
          }
        }
        return ApplePlatform.TVOS_SIMULATOR;
      case PlatformType.MACOS:
        return ApplePlatform.MACOS;
      case PlatformType.CATALYST:
        return ApplePlatform.CATALYST;
      default:
        throw new IllegalArgumentException("Unsupported platform type " + platformType);
    }
  }

  @Nullable
  @Override
  public String getXcodeVersionFlag() throws EvalException {
    return xcodeVersionFlag;
  }

  @Override
  public DottedVersion iosSdkVersionFlag() throws EvalException {
    return iosSdkVersionFlag;
  }

  @Override
  public DottedVersion macOsSdkVersionFlag() throws EvalException {
    return macOsSdkVersionFlag;
  }

  @Override
  public DottedVersion tvOsSdkVersionFlag() throws EvalException {
    return tvOsSdkVersionFlag;
  }

  @Override
  public DottedVersion watchOsSdkVersionFlag() throws EvalException {
    return watchOsSdkVersionFlag;
  }

  @Override
  public DottedVersion iosMinimumOsFlag() throws EvalException {
    return iosMinimumOsFlag;
  }

  @Override
  public DottedVersion macOsMinimumOsFlag() throws EvalException {
    return macosMinimumOsFlag;
  }

  @Override
  public DottedVersion tvOsMinimumOsFlag() throws EvalException {
    return tvosMinimumOsFlag;
  }

  @Override
  public DottedVersion watchOsMinimumOsFlag() throws EvalException {
    return watchosMinimumOsFlag;
  }

  @Override
  public boolean shouldPreferMutualXcode() throws EvalException {
    return preferMutualXcode;
  }

  @Override
  public boolean includeXcodeExecRequirementsFlag() throws EvalException {
    return includeXcodeExecRequirements;
  }

  /**
   * Returns the label of the xcode_config rule to use for resolving the exec system Xcode version.
   */
  @StarlarkConfigurationField(
      name = "xcode_config_label",
      doc = "Returns the target denoted by the value of the --xcode_version_config flag",
      defaultLabel = AppleCommandLineOptions.DEFAULT_XCODE_VERSION_CONFIG_LABEL,
      defaultInToolRepository = true)
  public Label getXcodeConfigLabel() {
    return xcodeConfigLabel;
  }

  @Override
  public void processForOutputPathMnemonic(Fragment.OutputDirectoriesContext ctx)
      throws Fragment.OutputDirectoriesContext.AddToMnemonicException {
    List<String> components = new ArrayList<>();
    if (!appleCpus.appleSplitCpu().isEmpty()) {
      components.add(applePlatformType);
      components.add(appleCpus.appleSplitCpu());

      if (options.getMinimumOsVersion() != null) {
        components.add("min" + options.getMinimumOsVersion());
      }
    }
    if (configurationDistinguisher != ConfigurationDistinguisher.UNKNOWN) {
      components.add(configurationDistinguisher.getFileSystemName());
    }

    if (!components.isEmpty()) {
      ctx.addToMnemonic(Joiner.on('-').join(components));
    }
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof AppleConfiguration that)) {
      return false;
    }
    return this.options.equals(that.options);
  }

  @Override
  public int hashCode() {
    return options.hashCode();
  }

  /**
   * Value used to avoid multiple configurations from conflicting. No two instances of this
   * transition may exist with the same value in a single Bazel invocation.
   */
  public enum ConfigurationDistinguisher implements StarlarkValue {
    UNKNOWN("unknown"),
    /** Distinguisher for {@code apple_binary} rule with "ios" platform_type. */
    APPLEBIN_IOS("applebin_ios"),
    /** Distinguisher for {@code apple_binary} rule with "visionos" platform_type. */
    APPLEBIN_VISIONOS("applebin_visionos"),
    /** Distinguisher for {@code apple_binary} rule with "watchos" platform_type. */
    APPLEBIN_WATCHOS("applebin_watchos"),
    /** Distinguisher for {@code apple_binary} rule with "tvos" platform_type. */
    APPLEBIN_TVOS("applebin_tvos"),
    /** Distinguisher for {@code apple_binary} rule with "macos" platform_type. */
    APPLEBIN_MACOS("applebin_macos"),
    /** Distinguisher for {@code apple_binary} rule with "catalyst" platform_type. */
    APPLEBIN_CATALYST("applebin_catalyst"),

    /**
     * Distinguisher for the apple crosstool configuration. We use "apl" for output directory names
     * instead of "apple_crosstool" to avoid oversized path names, which can be problematic on OSX.
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
  }
  // LINT.ThenChange(//src/main/starlark/builtins_bzl/common/objc/apple_configuration.bzl)
}
