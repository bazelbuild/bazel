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
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.util.CPU;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsClass;
import java.util.List;

/** Command-line options for building for Apple platforms. */
@OptionsClass
public abstract class AppleCommandLineOptions extends FragmentOptions {
  @Option(
      name = "xcode_version",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "If specified, uses Xcode of the given version for relevant build actions. "
              + "If unspecified, uses the executor default version of Xcode.")
  public abstract String getXcodeVersion();

  @Option(
      name = "ios_sdk_version",
      defaultValue = "null",
      converter = DottedVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "Specifies the version of the iOS SDK to use to build iOS applications. "
              + "If unspecified, uses the default iOS SDK version from 'xcode_version'.")
  public abstract DottedVersion.Option getIosSdkVersion();

  @Option(
      name = "watchos_sdk_version",
      defaultValue = "null",
      converter = DottedVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "Specifies the version of the watchOS SDK to use to build watchOS applications. "
              + "If unspecified, uses the default watchOS SDK version from 'xcode_version'.")
  public abstract DottedVersion.Option getWatchOsSdkVersion();

  @Option(
      name = "tvos_sdk_version",
      defaultValue = "null",
      converter = DottedVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "Specifies the version of the tvOS SDK to use to build tvOS applications. "
              + "If unspecified, uses the default tvOS SDK version from 'xcode_version'.")
  public abstract DottedVersion.Option getTvOsSdkVersion();

  @Option(
      name = "macos_sdk_version",
      defaultValue = "null",
      converter = DottedVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "Specifies the version of the macOS SDK to use to build macOS applications. "
              + "If unspecified, uses the default macOS SDK version from 'xcode_version'.")
  public abstract DottedVersion.Option getMacOsSdkVersion();

  @Option(
      name = "ios_minimum_os",
      defaultValue = "null",
      converter = DottedVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "Minimum compatible iOS version for target simulators and devices. "
              + "If unspecified, uses 'ios_sdk_version'.")
  public abstract DottedVersion.Option getIosMinimumOs();

  @Option(
      name = "watchos_minimum_os",
      defaultValue = "null",
      converter = DottedVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "Minimum compatible watchOS version for target simulators and devices. "
              + "If unspecified, uses 'watchos_sdk_version'.")
  public abstract DottedVersion.Option getWatchosMinimumOs();

  @Option(
      name = "tvos_minimum_os",
      defaultValue = "null",
      converter = DottedVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "Minimum compatible tvOS version for target simulators and devices. "
              + "If unspecified, uses 'tvos_sdk_version'.")
  public abstract DottedVersion.Option getTvosMinimumOs();

  @Option(
      name = "macos_minimum_os",
      defaultValue = "null",
      converter = DottedVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "Minimum compatible macOS version for targets. "
              + "If unspecified, uses 'macos_sdk_version'.")
  public abstract DottedVersion.Option getMacosMinimumOs();

  @Option(
      name = "host_macos_minimum_os",
      defaultValue = "null",
      converter = DottedVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "Minimum compatible macOS version for host targets. "
              + "If unspecified, uses 'macos_sdk_version'.")
  public abstract DottedVersion.Option getHostMacosMinimumOs();

  @Option(
      name = "experimental_prefer_mutual_xcode",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "If true, use the most recent Xcode that is available both locally and remotely. If"
              + " false, or if there are no mutual available versions, use the local Xcode version"
              + " selected via xcode-select.")
  public abstract boolean getPreferMutualXcode();

  // Tracked in #28081.
  @Option(
      name = "incompatible_remove_ctx_apple_fragment",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "When true, Apple build flags are defined with Apple rules (in BUIILD files) and"
              + " ctx.fragments.apple is undefined. This is a migration flag to move all Apple"
              + " flags from core Bazel to Apple rules.")
  public abstract boolean getDisableAppleFragment();

  @VisibleForTesting public static final String DEFAULT_IOS_SDK_VERSION = "8.4";
  @VisibleForTesting public static final String DEFAULT_WATCHOS_SDK_VERSION = "2.0";
  @VisibleForTesting public static final String DEFAULT_MACOS_SDK_VERSION = "10.11";
  @VisibleForTesting public static final String DEFAULT_TVOS_SDK_VERSION = "9.0";
  @VisibleForTesting static final String DEFAULT_IOS_CPU = "x86_64";

  /** The default visionOS CPU value. */
  public static final String DEFAULT_VISIONOS_CPU = "sim_arm64";

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
      name = "apple_platform_type",
      defaultValue = "macos",
      converter = PlatformTypeConverter.class,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help =
          "Don't set this value from the command line - it is derived from other flags and "
              + "configuration transitions derived from rule attributes")
  public abstract String getApplePlatformType();

  @Option(
      name = "apple_split_cpu",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help =
          "Don't set this value from the command line - it is derived from other flags and "
              + "configuration transitions derived from rule attributes")
  public abstract String getAppleSplitCpu();

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
  public abstract List<String> getIosMultiCpus();

  @Option(
      name = "visionos_cpus",
      allowMultiple = true,
      converter = CommaSeparatedOptionListConverter.class,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.LOADING_AND_ANALYSIS},
      help = "Comma-separated list of architectures for which to build Apple visionOS binaries.")
  public abstract List<String> getVisionosCpus();

  @Option(
      name = "watchos_cpus",
      allowMultiple = true,
      converter = CommaSeparatedOptionListConverter.class,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.LOADING_AND_ANALYSIS},
      help = "Comma-separated list of architectures for which to build Apple watchOS binaries.")
  public abstract List<String> getWatchosCpus();

  @Option(
      name = "tvos_cpus",
      allowMultiple = true,
      converter = CommaSeparatedOptionListConverter.class,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.LOADING_AND_ANALYSIS},
      help = "Comma-separated list of architectures for which to build Apple tvOS binaries.")
  public abstract List<String> getTvosCpus();

  @Option(
      name = "macos_cpus",
      allowMultiple = true,
      converter = CommaSeparatedOptionListConverter.class,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.LOADING_AND_ANALYSIS},
      help = "Comma-separated list of architectures for which to build Apple macOS binaries.")
  public abstract List<String> getMacosCpus();

  @Option(
      name = "xcode_version_config",
      defaultValue = "@bazel_tools//tools/cpp:host_xcodes",
      converter = LabelConverter.class,
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.LOADING_AND_ANALYSIS},
      help =
          "The label of the xcode_config rule to be used for selecting the Xcode version "
              + "in the build configuration.")
  public abstract Label getXcodeVersionConfig();

  @Option(
      name = "experimental_include_xcode_execution_requirements",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.EXECUTION
      },
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "If set, add a \"requires-xcode:{version}\" execution requirement to every Xcode action."
              + "  If the Xcode version has a hyphenated label,  also add a"
              + " \"requires-xcode-label:{version_label}\" execution requirement.")
  public abstract boolean getIncludeXcodeExecutionRequirements();

  /**
   * The default label of the build-wide {@code xcode_config} configuration rule. This can be
   * changed from the default using the {@code xcode_version_config} build flag.
   */
  // TODO(cparsons): Update all callers to reference the actual xcode_version_config flag value.
  @VisibleForTesting
  public static final String DEFAULT_XCODE_VERSION_CONFIG_LABEL = "//tools/objc:host_xcodes";

  @Option(
      name = "apple_platforms",
      converter = LabelListConverter.class,
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.LOADING_AND_ANALYSIS},
      help = "Comma-separated list of platforms to use when building Apple binaries.")
  public abstract List<Label> getApplePlatforms();

  @Option(
      name = "use_platforms_in_apple_crosstool_transition",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      help =
          "Makes apple_crosstool_transition fall back to using the value of `--platforms` flag"
              + " instead of legacy `--cpu` when needed.")
  public abstract boolean getUsePlatformsInAppleCrosstoolTransition();

  /** Returns whether the minimum OS version is explicitly set for the current platform. */
  public DottedVersion getMinimumOsVersion() {
    DottedVersion.Option option;
    switch (getApplePlatformType()) {
      case PlatformType.IOS:
      case PlatformType.CATALYST:
        option = getIosMinimumOs();
        break;
      case PlatformType.MACOS:
        option = getMacosMinimumOs();
        break;
      case PlatformType.TVOS:
        option = getTvosMinimumOs();
        break;
      case PlatformType.VISIONOS:
        // TODO: Replace with CppOptions.minimumOsVersion
        option = DottedVersion.option(DottedVersion.fromStringUnchecked("1.0"));
        break;
      case PlatformType.WATCHOS:
        option = getWatchosMinimumOs();
        break;
      default:
        throw new IllegalStateException();
    }

    return DottedVersion.maybeUnwrap(option);
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
