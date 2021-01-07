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
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.LabelConverter;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.starlarkbuildapi.apple.AppleBitcodeModeApi;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import net.starlark.java.eval.Printer;

/** Command-line options for building for Apple platforms. */
public class AppleCommandLineOptions extends FragmentOptions {
  @Option(
    name = "experimental_apple_mandatory_minimum_version",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.BUILD_FILE_SEMANTICS},
    help = "Whether Apple rules must have a mandatory minimum_os_version attribute."
  )
  // TODO(b/37096178): This flag should be default-on and then be removed.
  public boolean mandatoryMinimumVersion;

  @Option(
    name = "experimental_objc_provider_from_linked",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.BUILD_FILE_SEMANTICS},
    help =
        "Whether Apple rules which control linking should propagate objc provider at the top "
            + "level"
  )
  // TODO(b/32411441): This flag should be default-off and then be removed.
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
              + "If unspecified, uses default iOS SDK version from 'xcode_version'.")
  public DottedVersion.Option iosSdkVersion;

  @Option(
      name = "watchos_sdk_version",
      defaultValue = "null",
      converter = DottedVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "Specifies the version of the watchOS SDK to use to build watchOS applications. "
              + "If unspecified, uses default watchOS SDK version from 'xcode_version'.")
  public DottedVersion.Option watchOsSdkVersion;

  @Option(
      name = "tvos_sdk_version",
      defaultValue = "null",
      converter = DottedVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "Specifies the version of the tvOS SDK to use to build tvOS applications. "
              + "If unspecified, uses default tvOS SDK version from 'xcode_version'.")
  public DottedVersion.Option tvOsSdkVersion;

  @Option(
      name = "macos_sdk_version",
      defaultValue = "null",
      converter = DottedVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "Specifies the version of the macOS SDK to use to build macOS applications. "
              + "If unspecified, uses default macOS SDK version from 'xcode_version'.")
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
  @VisibleForTesting public static final String DEFAULT_MACOS_SDK_VERSION = "10.10";
  @VisibleForTesting public static final String DEFAULT_TVOS_SDK_VERSION = "9.0";
  @VisibleForTesting static final String DEFAULT_IOS_CPU = "x86_64";

  /** The default watchos CPU value. */
  public static final String DEFAULT_WATCHOS_CPU = "i386";

  /** The default tvOS CPU value. */
  public static final String DEFAULT_TVOS_CPU = "x86_64";

  /** The default macOS CPU value. */
  public static final String DEFAULT_MACOS_CPU = "x86_64";

  /** The default Catalyst CPU value. */
  public static final String DEFAULT_CATALYST_CPU = "x86_64";

  @Option(
    name = "ios_cpu",
    defaultValue = DEFAULT_IOS_CPU,
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.NO_OP},
    metadataTags = {OptionMetadataTag.DEPRECATED},
    help = "Specifies to target CPU of iOS compilation."
  )
  public String iosCpu;

  @Option(
    name = "apple_compiler",
    defaultValue = "null",
    documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
    effectTags = {
      OptionEffectTag.AFFECTS_OUTPUTS,
      OptionEffectTag.LOADING_AND_ANALYSIS,
      OptionEffectTag.LOSES_INCREMENTAL_STATE,
    },
    help = "The Apple target compiler. Useful for selecting variants of a toolchain "
               + "(e.g. xcode-beta)."
  )
  public String cppCompiler;

  @Option(
    name = "apple_grte_top",
    defaultValue = "null",
    converter = LabelConverter.class,
    documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
    effectTags = {
      OptionEffectTag.CHANGES_INPUTS,
      OptionEffectTag.LOADING_AND_ANALYSIS,
      OptionEffectTag.LOSES_INCREMENTAL_STATE,
    },
    help = "The Apple target grte_top."
  )
  public Label appleLibcTop;

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
      defaultValue = "MACOS",
      converter = PlatformTypeConverter.class,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help =
          "Don't set this value from the command line - it is derived from other flags and "
              + "configuration transitions derived from rule attributes")
  public PlatformType applePlatformType;

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
  @Option(
    name = "apple configuration distinguisher",
    defaultValue = "UNKNOWN",
    converter = ConfigurationDistinguisherConverter.class,
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
    metadataTags = {OptionMetadataTag.INTERNAL}
  )
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
    defaultValue = "@local_config_xcode//:host_xcodes",
    converter = LabelConverter.class,
    documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
    effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.LOADING_AND_ANALYSIS},
    help =
        "The label of the xcode_config rule to be used for selecting the Xcode version "
            + "in the build configuration."
  )
  public Label xcodeVersionConfig;

  /**
   * The default label of the build-wide {@code xcode_config} configuration rule. This can be
   * changed from the default using the {@code xcode_version_config} build flag.
   */
  // TODO(cparsons): Update all callers to reference the actual xcode_version_config flag value.
  @VisibleForTesting
  public static final String DEFAULT_XCODE_VERSION_CONFIG_LABEL = "//tools/objc:host_xcodes";

  @Option(
      name = "apple_bitcode",
      allowMultiple = true,
      converter = AppleBitcodeConverter.class,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "Specify the Apple bitcode mode for compile steps targeting device architectures. Values"
              + " are of the form '[platform=]mode', where the platform (which must be 'ios',"
              + " 'macos', 'tvos', or 'watchos') is optional. If provided, the bitcode mode is"
              + " applied for that platform specifically; if omitted, it is applied for all"
              + " platforms. The mode must be 'none', 'embedded_markers', or 'embedded'. This"
              + " option may be provided multiple times.")
  public List<Map.Entry<ApplePlatform.PlatformType, AppleBitcodeMode>> appleBitcodeMode;

  /** Returns whether the minimum OS version is explicitly set for the current platform. */
  public DottedVersion getMinimumOsVersion() {
    DottedVersion.Option option;
    switch (applePlatformType) {
      case IOS:
      case CATALYST:
        option = iosMinimumOs;
        break;
      case MACOS:
        option = macosMinimumOs;
        break;
      case TVOS:
        option = tvosMinimumOs;
        break;
      case WATCHOS:
        option = watchosMinimumOs;
        break;
      default:
        throw new IllegalStateException();
    }

    return DottedVersion.maybeUnwrap(option);
  }

  /**
   * Represents the Apple Bitcode mode for compilation steps.
   *
   * <p>Bitcode is an intermediate representation of a compiled program. For many platforms, Apple
   * requires app submissions to contain bitcode in order to be uploaded to the app store.
   *
   * <p>This is a build-wide value, as bitcode mode needs to be consistent among a target and its
   * compiled dependencies.
   */
  @Immutable
  public enum AppleBitcodeMode implements AppleBitcodeModeApi {

    /** Do not compile bitcode. */
    NONE("none", ImmutableList.<String>of()),
    /**
     * Compile the minimal set of bitcode markers. This is often the best option for developer/debug
     * builds.
     */
    EMBEDDED_MARKERS("embedded_markers", ImmutableList.of("bitcode_embedded_markers")),
    /** Fully embed bitcode in compiled files. This is often the best option for release builds. */
    EMBEDDED("embedded", ImmutableList.of("bitcode_embedded"));

    private final String mode;
    private final ImmutableList<String> featureNames;

    private AppleBitcodeMode(String mode, ImmutableList<String> featureNames) {
      this.mode = mode;
      this.featureNames = featureNames;
    }

    @Override
    public boolean isImmutable() {
      return true; // immutable and Starlark-hashable
    }

    @Override
    public String toString() {
      return mode;
    }

    @Override
    public void repr(Printer printer) {
      printer.append(mode);
    }

    /** Returns the names of any crosstool features that correspond to this bitcode mode. */
    public ImmutableList<String> getFeatureNames() {
      return featureNames;
    }

    /** Converts to {@link AppleBitcodeMode}. */
    public static class Converter extends EnumConverter<AppleBitcodeMode> {
      public Converter() {
        super(AppleBitcodeMode.class, "apple bitcode mode");
      }
    }
  }

  @Override
  public FragmentOptions getHost() {
    AppleCommandLineOptions host = (AppleCommandLineOptions) super.getHost();

    // Set options needed in the host configuration.
    host.xcodeVersionConfig = xcodeVersionConfig;
    host.xcodeVersion = xcodeVersion;
    host.iosSdkVersion = iosSdkVersion;
    host.watchOsSdkVersion = watchOsSdkVersion;
    host.tvOsSdkVersion = tvOsSdkVersion;
    host.macOsSdkVersion = macOsSdkVersion;
    host.appleBitcodeMode = appleBitcodeMode;
    // The host apple platform type will always be MACOS, as no other apple platform type can
    // currently execute build actions. If that were the case, a host_apple_platform_type flag might
    // be needed.
    host.applePlatformType = PlatformType.MACOS;
    host.configurationDistinguisher = ConfigurationDistinguisher.UNKNOWN;

    return host;
  }

  void serialize(SerializationContext context, CodedOutputStream out)
      throws IOException, SerializationException {
    context.serialize(this, out);
  }

  static AppleCommandLineOptions deserialize(DeserializationContext context, CodedInputStream in)
      throws IOException, SerializationException {
    return context.deserialize(in);
  }

  /** Converter for the Apple configuration distinguisher. */
  public static final class ConfigurationDistinguisherConverter
      extends EnumConverter<ConfigurationDistinguisher> {
    public ConfigurationDistinguisherConverter() {
      super(ConfigurationDistinguisher.class, "Apple rule configuration distinguisher");
    }
  }

  /** Flag converter for {@link PlatformType}. */
  public static final class PlatformTypeConverter extends EnumConverter<PlatformType> {
    public PlatformTypeConverter() {
      super(PlatformType.class, "Apple platform type");
    }
  }
}
