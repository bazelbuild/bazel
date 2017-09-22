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

import static com.google.devtools.build.lib.skyframe.serialization.SerializationCommonUtils.STRING_LIST_CODEC;
import static com.google.devtools.build.lib.skyframe.serialization.SerializationCommonUtils.deserializeNullable;
import static com.google.devtools.build.lib.skyframe.serialization.SerializationCommonUtils.serializeNullable;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.DefaultLabelConverter;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.LabelConverter;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.skyframe.serialization.EnumCodec;
import com.google.devtools.build.lib.skyframe.serialization.FastStringCodec;
import com.google.devtools.build.lib.skyframe.serialization.LabelCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
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

/**
 * Command-line options for building for Apple platforms.
 */
public class AppleCommandLineOptions extends FragmentOptions {

  @Option(
    name = "experimental_apple_mandatory_minimum_version",
    defaultValue = "false",
    category = "experimental",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags =  { OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.BUILD_FILE_SEMANTICS },
    help = "Whether Apple rules must have a mandatory minimum_os_version attribute."
  )
  // TODO(b/37096178): This flag should be default-on and then be removed.
  public boolean mandatoryMinimumVersion;

  @Option(
    name = "experimental_objc_provider_from_linked",
    defaultValue = "true",
    category = "experimental",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags =  { OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.BUILD_FILE_SEMANTICS },
    help = "Whether Apple rules which control linking should propagate objc provider at the top "
        + "level"
  )
  // TODO(b/32411441): This flag should be default-off and then be removed.
  public boolean objcProviderFromLinked;

  @Option(
    name = "xcode_version",
    defaultValue = "null",
    category = "build",
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
    category = "build",
    documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
    effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
    help = "Specifies the version of the iOS SDK to use to build iOS applications."
  )
  public DottedVersion iosSdkVersion;

  @Option(
    name = "watchos_sdk_version",
    defaultValue = "null",
    converter = DottedVersionConverter.class,
    category = "build",
    documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
    effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
    help = "Specifies the version of the watchOS SDK to use to build watchOS applications."
  )
  public DottedVersion watchOsSdkVersion;

  @Option(
    name = "tvos_sdk_version",
    defaultValue = "null",
    converter = DottedVersionConverter.class,
    category = "build",
    documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
    effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
    help = "Specifies the version of the tvOS SDK to use to build tvOS applications."
  )
  public DottedVersion tvOsSdkVersion;

  @Option(
    name = "macos_sdk_version",
    defaultValue = "null",
    converter = DottedVersionConverter.class,
    category = "build",
    documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
    effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
    help = "Specifies the version of the macOS SDK to use to build macOS applications."
  )
  public DottedVersion macOsSdkVersion;

  @Option(
    name = "ios_minimum_os",
    defaultValue = "null",
    category = "flags",
    converter = DottedVersionConverter.class,
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
    help = "Minimum compatible iOS version for target simulators and devices."
  )
  public DottedVersion iosMinimumOs;

  @Option(
    name = "watchos_minimum_os",
    defaultValue = "null",
    category = "flags",
    converter = DottedVersionConverter.class,
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
    help = "Minimum compatible watchOS version for target simulators and devices."
  )
  public DottedVersion watchosMinimumOs;

  @Option(
    name = "tvos_minimum_os",
    defaultValue = "null",
    category = "flags",
    converter = DottedVersionConverter.class,
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
    help = "Minimum compatible tvOS version for target simulators and devices."
  )
  public DottedVersion tvosMinimumOs;

  @Option(
    name = "macos_minimum_os",
    defaultValue = "null",
    category = "flags",
    converter = DottedVersionConverter.class,
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
    help = "Minimum compatible macOS version for targets."
  )
  public DottedVersion macosMinimumOs;

  @VisibleForTesting public static final String DEFAULT_IOS_SDK_VERSION = "8.4";
  @VisibleForTesting public static final String DEFAULT_WATCHOS_SDK_VERSION = "2.0";
  @VisibleForTesting public static final String DEFAULT_MACOS_SDK_VERSION = "10.10";
  @VisibleForTesting public static final String DEFAULT_TVOS_SDK_VERSION = "9.0";
  @VisibleForTesting static final String DEFAULT_IOS_CPU = "x86_64";

  /**
   * The default watchos CPU value.
   */
  public static final String DEFAULT_WATCHOS_CPU = "i386";

  /**
   * The default tvOS CPU value.
   */
  public static final String DEFAULT_TVOS_CPU = "x86_64";

  /**
   * The default macOS CPU value.
   */
  public static final String DEFAULT_MACOS_CPU = "x86_64";

  @Option(
    name = "ios_cpu",
    defaultValue = DEFAULT_IOS_CPU,
    category = "build",
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.NO_OP},
    metadataTags = {OptionMetadataTag.DEPRECATED},
    help = "Specifies to target CPU of iOS compilation."
  )
  public String iosCpu;

  @Option(
    name = "apple_crosstool_top",
    defaultValue = "@bazel_tools//tools/cpp:toolchain",
    category = "version",
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
    defaultValue = "IOS",
    converter = PlatformTypeConverter.class,
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
    help =
        "Don't set this value from the command line - it is derived from other flags and "
            + "configuration transitions derived from rule attributes"
  )
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
    converter = CommaSeparatedOptionListConverter.class,
    defaultValue = "",
    category = "flags",
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.LOADING_AND_ANALYSIS},
    help =
        "Comma-separated list of architectures to build an ios_application with. The result "
            + "is a universal binary containing all specified architectures."
  )
  public List<String> iosMultiCpus;

  @Option(
    name = "watchos_cpus",
    converter = CommaSeparatedOptionListConverter.class,
    defaultValue = DEFAULT_WATCHOS_CPU,
    category = "flags",
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.LOADING_AND_ANALYSIS},
    help = "Comma-separated list of architectures for which to build Apple watchOS binaries."
  )
  public List<String> watchosCpus;

  @Option(
    name = "tvos_cpus",
    converter = CommaSeparatedOptionListConverter.class,
    defaultValue = DEFAULT_TVOS_CPU,
    category = "flags",
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.LOADING_AND_ANALYSIS},
    help = "Comma-separated list of architectures for which to build Apple tvOS binaries."
  )
  public List<String> tvosCpus;

  @Option(
    name = "macos_cpus",
    converter = CommaSeparatedOptionListConverter.class,
    defaultValue = DEFAULT_MACOS_CPU,
    category = "flags",
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.LOADING_AND_ANALYSIS},
    help = "Comma-separated list of architectures for which to build Apple macOS binaries."
  )
  public List<String> macosCpus;

  @Option(
    name = "default_ios_provisioning_profile",
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.SIGNING,
    effectTags = {OptionEffectTag.CHANGES_INPUTS},
    converter = DefaultProvisioningProfileConverter.class
  )
  public Label defaultProvisioningProfile;

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
  static final String DEFAULT_XCODE_VERSION_CONFIG_LABEL = "//tools/objc:host_xcodes";

  /** Converter for --default_ios_provisioning_profile. */
  public static class DefaultProvisioningProfileConverter extends DefaultLabelConverter {
    public DefaultProvisioningProfileConverter() {
      super("//tools/objc:default_provisioning_profile");
    }
  }

  @Option(
    name = "xcode_toolchain",
    defaultValue = "null",
    category = "flags",
    documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
    effectTags = {OptionEffectTag.ACTION_OPTIONS},
    help =
        "The identifier of an Xcode toolchain to use for builds. Currently only the toolchains "
            + "that ship with Xcode are supported. For example, in addition to the default "
            + "toolchain Xcode 8 has 'com.apple.dt.toolchain.Swift_2_3' which can be used for "
            + "building legacy Swift code."
  )
  public String xcodeToolchain;

  @Option(
    name = "apple_bitcode",
    converter = AppleBitcodeMode.Converter.class,
    // TODO(blaze-team): Default to embedded_markers when fully implemented.
    defaultValue = "none",
    category = "flags",
    documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
    effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE},
    help =
        "Specify the Apple bitcode mode for compile steps. "
            + "Values: 'none', 'embedded_markers', 'embedded'."
  )
  public AppleBitcodeMode appleBitcodeMode;

  @Option(
    name = "apple_crosstool_transition",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.CHANGES_INPUTS},
    metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
    help = "If true, the apple crosstool is used for all apple rules."
  )
  public boolean enableAppleCrosstoolTransition;

  @Option(
    name = "target_uses_apple_crosstool",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
    help = "If true, this target uses the apple crosstool.  Do not set this flag manually."
  )
  public boolean targetUsesAppleCrosstool;

  /**
   * Returns the architecture implied by these options.
   *
   * <p> In contexts in which a configuration instance is present, prefer
   * {@link AppleConfiguration#getSingleArchitecture}.
   */
  public String getSingleArchitecture() {
    if (!Strings.isNullOrEmpty(appleSplitCpu)) {
      return appleSplitCpu;
    }
    switch (applePlatformType) {
      case IOS:
        if (!iosMultiCpus.isEmpty()) {
          return iosMultiCpus.get(0);
        } else {
          return iosCpu;
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
   * Represents the Apple Bitcode mode for compilation steps.
   *
   * <p>Bitcode is an intermediate representation of a compiled program. For many platforms, Apple
   * requires app submissions to contain bitcode in order to be uploaded to the app store.
   *
   * <p>This is a build-wide value, as bitcode mode needs to be consistent among a target and its
   * compiled dependencies.
   */
  @SkylarkModule(
    name = "apple_bitcode_mode",
    category = SkylarkModuleCategory.NONE,
    doc = "The Bitcode mode to use when compiling Objective-C and Swift code on Apple platforms. "
        + "Possible values are:<br><ul>"
        + "<li><code>'none'</code></li>"
        + "<li><code>'embedded'</code></li>"
        + "<li><code>'embedded_markers'</code></li>"
        + "</ul>"
  )
  @Immutable
  public enum AppleBitcodeMode implements SkylarkValue {

    /** Do not compile bitcode. */
    NONE("none", ImmutableList.<String>of()),
    /**
     * Compile the minimal set of bitcode markers. This is often the best option for developer/debug
     * builds.
     */
    EMBEDDED_MARKERS(
        "embedded_markers", ImmutableList.of("bitcode_embedded_markers"), "-fembed-bitcode-marker"),
    /** Fully embed bitcode in compiled files. This is often the best option for release builds. */
    EMBEDDED("embedded", ImmutableList.of("bitcode_embedded"), "-fembed-bitcode");

    private final String mode;
    private final ImmutableList<String> featureNames;
    private final ImmutableList<String> clangFlags;

    private AppleBitcodeMode(
        String mode, ImmutableList<String> featureNames, String... clangFlags) {
      this.mode = mode;
      this.featureNames = featureNames;
      this.clangFlags = ImmutableList.copyOf(clangFlags);
    }

    @Override
    public String toString() {
      return mode;
    }

    @Override
    public void repr(SkylarkPrinter printer) {
      printer.append(mode);
    }

    /** Returns the names of any crosstool features that correspond to this bitcode mode. */
    public ImmutableList<String> getFeatureNames() {
      return featureNames;
    }

    /**
     * Returns the flags that should be added to compile and link actions to use this
     * bitcode setting.
     */
    public ImmutableList<String> getCompileAndLinkFlags() {
      return clangFlags;
    }

    /**
     * Converts to {@link AppleBitcodeMode}.
     */
    public static class Converter extends EnumConverter<AppleBitcodeMode> {
      public Converter() {
        super(AppleBitcodeMode.class, "apple bitcode mode");
      }
    }

    static final EnumCodec<AppleBitcodeMode> CODEC = new EnumCodec<>(AppleBitcodeMode.class);
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

    return host;
  }

  void serialize(CodedOutputStream out) throws IOException, SerializationException {
    out.writeBoolNoTag(mandatoryMinimumVersion);
    out.writeBoolNoTag(objcProviderFromLinked);
    serializeNullable(xcodeVersion, out, FastStringCodec.INSTANCE);
    serializeNullable(iosSdkVersion, out, DottedVersion.CODEC);
    serializeNullable(watchOsSdkVersion, out, DottedVersion.CODEC);
    serializeNullable(tvOsSdkVersion, out, DottedVersion.CODEC);
    serializeNullable(macOsSdkVersion, out, DottedVersion.CODEC);
    serializeNullable(iosMinimumOs, out, DottedVersion.CODEC);
    serializeNullable(watchosMinimumOs, out, DottedVersion.CODEC);
    serializeNullable(tvosMinimumOs, out, DottedVersion.CODEC);
    serializeNullable(macosMinimumOs, out, DottedVersion.CODEC);
    FastStringCodec.INSTANCE.serialize(iosCpu, out);
    LabelCodec.INSTANCE.serialize(appleCrosstoolTop, out);
    PlatformType.CODEC.serialize(applePlatformType, out);
    FastStringCodec.INSTANCE.serialize(appleSplitCpu, out);
    ConfigurationDistinguisher.CODEC.serialize(configurationDistinguisher, out);
    STRING_LIST_CODEC.serialize((ImmutableList<String>) iosMultiCpus, out);
    STRING_LIST_CODEC.serialize((ImmutableList<String>) watchosCpus, out);
    STRING_LIST_CODEC.serialize((ImmutableList<String>) tvosCpus, out);
    STRING_LIST_CODEC.serialize((ImmutableList<String>) macosCpus, out);
    LabelCodec.INSTANCE.serialize(defaultProvisioningProfile, out);
    LabelCodec.INSTANCE.serialize(xcodeVersionConfig, out);
    serializeNullable(xcodeToolchain, out, FastStringCodec.INSTANCE);
    AppleBitcodeMode.CODEC.serialize(appleBitcodeMode, out);
    out.writeBoolNoTag(enableAppleCrosstoolTransition);
    out.writeBoolNoTag(targetUsesAppleCrosstool);
  }

  static AppleCommandLineOptions deserialize(CodedInputStream in)
      throws IOException, SerializationException {
    AppleCommandLineOptions result = new AppleCommandLineOptions();
    result.mandatoryMinimumVersion = in.readBool();
    result.objcProviderFromLinked = in.readBool();
    result.xcodeVersion = deserializeNullable(in, FastStringCodec.INSTANCE);
    result.iosSdkVersion = deserializeNullable(in, DottedVersion.CODEC);
    result.watchOsSdkVersion = deserializeNullable(in, DottedVersion.CODEC);
    result.tvOsSdkVersion = deserializeNullable(in, DottedVersion.CODEC);
    result.macOsSdkVersion = deserializeNullable(in, DottedVersion.CODEC);
    result.iosMinimumOs = deserializeNullable(in, DottedVersion.CODEC);
    result.watchosMinimumOs = deserializeNullable(in, DottedVersion.CODEC);
    result.tvosMinimumOs = deserializeNullable(in, DottedVersion.CODEC);
    result.macosMinimumOs = deserializeNullable(in, DottedVersion.CODEC);
    result.iosCpu = FastStringCodec.INSTANCE.deserialize(in);
    result.appleCrosstoolTop = LabelCodec.INSTANCE.deserialize(in);
    result.applePlatformType = PlatformType.CODEC.deserialize(in);
    result.appleSplitCpu = FastStringCodec.INSTANCE.deserialize(in);
    result.configurationDistinguisher = ConfigurationDistinguisher.CODEC.deserialize(in);
    result.iosMultiCpus = STRING_LIST_CODEC.deserialize(in);
    result.watchosCpus = STRING_LIST_CODEC.deserialize(in);
    result.tvosCpus = STRING_LIST_CODEC.deserialize(in);
    result.macosCpus = STRING_LIST_CODEC.deserialize(in);
    result.defaultProvisioningProfile = LabelCodec.INSTANCE.deserialize(in);
    result.xcodeVersionConfig = LabelCodec.INSTANCE.deserialize(in);
    result.xcodeToolchain = deserializeNullable(in, FastStringCodec.INSTANCE);
    result.appleBitcodeMode = AppleBitcodeMode.CODEC.deserialize(in);
    result.enableAppleCrosstoolTransition = in.readBool();
    result.targetUsesAppleCrosstool = in.readBool();
    return result;
  }

  /** Converter for the Apple configuration distinguisher. */
  public static final class ConfigurationDistinguisherConverter
      extends EnumConverter<ConfigurationDistinguisher> {
    public ConfigurationDistinguisherConverter() {
      super(ConfigurationDistinguisher.class, "Apple rule configuration distinguisher");
    }
  }

  /** Flag converter for {@link PlatformType}. */
  public static final class PlatformTypeConverter
      extends EnumConverter<PlatformType> {
    public PlatformTypeConverter() {
      super(PlatformType.class, "Apple platform type");
    }
  }
}
