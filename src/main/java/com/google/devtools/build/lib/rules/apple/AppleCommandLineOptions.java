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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Ascii;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.LabelConverter;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.LabelListConverter;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.util.CPU;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import java.util.List;

/** Command-line options for building for Apple platforms. */
public class AppleCommandLineOptions extends FragmentOptions {
  @Option(
      name = "experimental_objc_provider_from_linked",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.NO_OP},
      help =
          "No-op. Kept here for backwards compatibility. This field will be removed in a "
              + "future release.")
  // TODO(b/32411441): This flag should be removed.
  public boolean objcProviderFromLinked;

  @Option(
    name = "xcode_version",
    defaultValue = "null",
    documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
    effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
    help =
        "If specified, uses Xcode of the given version for relevant build actions. "
            + "If unspecified, uses the executor default version of Xcode."
  )
  public String xcodeVersion;

  @Option(
      name = "ios_sdk_version",
      defaultValue = "null",
      converter = DottedVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "Specifies the version of the iOS SDK to use to build iOS applications. "
              + "If unspecified, uses the default iOS SDK version from 'xcode_version'.")
  public DottedVersion.Option iosSdkVersion;

  @Option(
      name = "watchos_sdk_version",
      defaultValue = "null",
      converter = DottedVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "Specifies the version of the watchOS SDK to use to build watchOS applications. "
              + "If unspecified, uses the default watchOS SDK version from 'xcode_version'.")
  public DottedVersion.Option watchOsSdkVersion;

  @Option(
      name = "tvos_sdk_version",
      defaultValue = "null",
      converter = DottedVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "Specifies the version of the tvOS SDK to use to build tvOS applications. "
              + "If unspecified, uses the default tvOS SDK version from 'xcode_version'.")
  public DottedVersion.Option tvOsSdkVersion;

  @Option(
      name = "macos_sdk_version",
      defaultValue = "null",
      converter = DottedVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "Specifies the version of the macOS SDK to use to build macOS applications. "
              + "If unspecified, uses the default macOS SDK version from 'xcode_version'.")
  public DottedVersion.Option macOsSdkVersion;

  @Option(
      name = "ios_minimum_os",
      defaultValue = "null",
      converter = DottedVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "Minimum compatible iOS version for target simulators and devices. "
              + "If unspecified, uses 'ios_sdk_version'.")
  public DottedVersion.Option iosMinimumOs;

  @Option(
      name = "watchos_minimum_os",
      defaultValue = "null",
      converter = DottedVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "Minimum compatible watchOS version for target simulators and devices. "
              + "If unspecified, uses 'watchos_sdk_version'.")
  public DottedVersion.Option watchosMinimumOs;

  @Option(
      name = "tvos_minimum_os",
      defaultValue = "null",
      converter = DottedVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "Minimum compatible tvOS version for target simulators and devices. "
              + "If unspecified, uses 'tvos_sdk_version'.")
  public DottedVersion.Option tvosMinimumOs;

  @Option(
      name = "macos_minimum_os",
      defaultValue = "null",
      converter = DottedVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "Minimum compatible macOS version for targets. "
              + "If unspecified, uses 'macos_sdk_version'.")
  public DottedVersion.Option macosMinimumOs;

  @Option(
      name = "host_macos_minimum_os",
      defaultValue = "null",
      converter = DottedVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "Minimum compatible macOS version for host targets. "
              + "If unspecified, uses 'macos_sdk_version'.")
  public DottedVersion.Option hostMacosMinimumOs;

  @Option(
      name = "experimental_prefer_mutual_xcode",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "If true, use the most recent Xcode that is available both locally and remotely. If"
              + " false, or if there are no mutual available versions, use the local Xcode version"
              + " selected via xcode-select.")
  public boolean preferMutualXcode;

  @VisibleForTesting public static final String DEFAULT_IOS_SDK_VERSION = "8.4";
  @VisibleForTesting public static final String DEFAULT_WATCHOS_SDK_VERSION = "2.0";
  @VisibleForTesting public static final String DEFAULT_MACOS_SDK_VERSION = "10.11";
  @VisibleForTesting public static final String DEFAULT_TVOS_SDK_VERSION = "9.0";
  @VisibleForTesting static final String DEFAULT_IOS_CPU = "x86_64";

  /** The default visionOS CPU value. */
  public static final String DEFAULT_VISIONOS_CPU =
      CPU.getCurrent() == CPU.AARCH64 ? "sim_arm64" : "x86_64";

  /** The default watchos CPU value. */
  public static final String DEFAULT_WATCHOS_CPU =
      CPU.getCurrent() == CPU.AARCH64 ? "arm64" : "x86_64";

  /** The default tvOS CPU value. */
  public static final String DEFAULT_TVOS_CPU =
      CPU.getCurrent() == CPU.AARCH64 ? "sim_arm64" : "x86_64";

  /** The default macOS CPU value. */
  public static final String DEFAULT_MACOS_CPU =
      CPU.getCurrent() == CPU.AARCH64 ? "arm64" : "x86_64";

  /** The default Catalyst CPU value. */
  public static final String DEFAULT_CATALYST_CPU = "x86_64";

  @Option(
    name = "apple_crosstool_top",
    defaultValue = "@bazel_tools//tools/cpp:toolchain",
    converter = LabelConverter.class,
    documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
    effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.CHANGES_INPUTS},
    help =
        "The label of the crosstool package to be used in Apple and Objc rules and their"
            + " dependencies."
  )
  public Label appleCrosstoolTop;

  @Option(
      name = "apple_platform_type",
      defaultValue = "macos",
      converter = PlatformTypeConverter.class,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help =
          "Don't set this value from the command line - it is derived from other flags and "
              + "configuration transitions derived from rule attributes")
  public String applePlatformType;

  @Option(
    name = "apple_split_cpu",
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
    help =
        "Don't set this value from the command line - it is derived from other flags and "
            + "configuration transitions derived from rule attributes"
  )
  public String appleSplitCpu;

  // This option exists because two configurations are not allowed to have the same cache key
  // (partially derived from options). Since we have multiple transitions that may result in the
  // same configuration values at runtime we need an artificial way to distinguish between them.
  // This option must only be set by those transitions for this purpose.
  // TODO(bazel-team): Remove this once we have dynamic configurations but make sure that different
  // configurations (e.g. by min os version) always use different output paths.
  // TODO(blaze-configurability-team): Deprecate this when legacy output directory scheme is gone.
  @Option(
      name = "apple configuration distinguisher",
      defaultValue = "UNKNOWN",
      converter = ConfigurationDistinguisherConverter.class,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      metadataTags = {OptionMetadataTag.INTERNAL})
  public ConfigurationDistinguisher configurationDistinguisher;

  @Option(
      name = "ios_multi_cpus",
      allowMultiple = true,
      converter = CommaSeparatedOptionListConverter.class,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.LOADING_AND_ANALYSIS},
      help =
          "Comma-separated list of architectures to build an ios_application with. The result "
              + "is a universal binary containing all specified architectures.")
  public List<String> iosMultiCpus;

  @Option(
      name = "visionos_cpus",
      allowMultiple = true,
      converter = CommaSeparatedOptionListConverter.class,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.LOADING_AND_ANALYSIS},
      help = "Comma-separated list of architectures for which to build Apple visionOS binaries.")
  public List<String> visionosCpus;

  @Option(
      name = "watchos_cpus",
      allowMultiple = true,
      converter = CommaSeparatedOptionListConverter.class,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.LOADING_AND_ANALYSIS},
      help = "Comma-separated list of architectures for which to build Apple watchOS binaries.")
  public List<String> watchosCpus;

  @Option(
      name = "tvos_cpus",
      allowMultiple = true,
      converter = CommaSeparatedOptionListConverter.class,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.LOADING_AND_ANALYSIS},
      help = "Comma-separated list of architectures for which to build Apple tvOS binaries.")
  public List<String> tvosCpus;

  @Option(
      name = "macos_cpus",
      allowMultiple = true,
      converter = CommaSeparatedOptionListConverter.class,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.LOADING_AND_ANALYSIS},
      help = "Comma-separated list of architectures for which to build Apple macOS binaries.")
  public List<String> macosCpus;

  @Option(
      name = "catalyst_cpus",
      allowMultiple = true,
      converter = CommaSeparatedOptionListConverter.class,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.LOADING_AND_ANALYSIS},
      help = "Comma-separated list of architectures for which to build Apple Catalyst binaries.")
  public List<String> catalystCpus;

  @Option(
      name = "xcode_version_config",
      defaultValue = "@bazel_tools//tools/cpp:host_xcodes",
      converter = LabelConverter.class,
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.LOADING_AND_ANALYSIS},
      help =
          "The label of the xcode_config rule to be used for selecting the Xcode version "
              + "in the build configuration.")
  public Label xcodeVersionConfig;

  @Option(
      name = "experimental_include_xcode_execution_requirements",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.EXECUTION
      },
      help =
          "If set, add a \"requires-xcode:{version}\" execution requirement to every Xcode action."
              + "  If the Xcode version has a hyphenated label,  also add a"
              + " \"requires-xcode-label:{version_label}\" execution requirement.")
  public boolean includeXcodeExecutionRequirements;

  /**
   * The default label of the build-wide {@code xcode_config} configuration rule. This can be
   * changed from the default using the {@code xcode_version_config} build flag.
   */
  // TODO(cparsons): Update all callers to reference the actual xcode_version_config flag value.
  @VisibleForTesting
  public static final String DEFAULT_XCODE_VERSION_CONFIG_LABEL = "//tools/objc:host_xcodes";

  @Option(
      name = "incompatible_enable_apple_toolchain_resolution",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "Use toolchain resolution to select the Apple SDK for apple rules (Starlark and native)")
  public boolean incompatibleUseToolchainResolution;

  @Option(
      name = "apple_platforms",
      converter = LabelListConverter.class,
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.LOADING_AND_ANALYSIS},
      help = "Comma-separated list of platforms to use when building Apple binaries.")
  public List<Label> applePlatforms;

  /** Returns whether the minimum OS version is explicitly set for the current platform. */
  public DottedVersion getMinimumOsVersion() {
    DottedVersion.Option option;
    switch (applePlatformType) {
      case PlatformType.IOS:
      case PlatformType.CATALYST:
        option = iosMinimumOs;
        break;
      case PlatformType.MACOS:
        option = macosMinimumOs;
        break;
      case PlatformType.TVOS:
        option = tvosMinimumOs;
        break;
      case PlatformType.VISIONOS:
        // TODO: Replace with CppOptions.minimumOsVersion
        option = DottedVersion.option(DottedVersion.fromStringUnchecked("1.0"));
        break;
      case PlatformType.WATCHOS:
        option = watchosMinimumOs;
        break;
      default:
        throw new IllegalStateException();
    }

    return DottedVersion.maybeUnwrap(option);
  }

  /** Converter for the Apple configuration distinguisher. */
  public static final class ConfigurationDistinguisherConverter
      extends EnumConverter<ConfigurationDistinguisher> {
    public ConfigurationDistinguisherConverter() {
      super(ConfigurationDistinguisher.class, "Apple rule configuration distinguisher");
    }
  }

  /** Flag converter for PlatformType string flag, just converting to lowercase. */
  public static final class PlatformTypeConverter extends Converter.Contextless<String> {
    public PlatformTypeConverter() {}

    @Override
    public String convert(String input) {
      return Ascii.toLowerCase(input);
    }

    @Override
    public final String getTypeDescription() {
      return "a string";
    }
  }
}
