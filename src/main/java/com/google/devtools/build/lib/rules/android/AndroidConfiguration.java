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
package com.google.devtools.build.lib.rules.android;

import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.EmptyToNullLabelConverter;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.LabelConverter;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.skylark.annotations.SkylarkConfigurationField;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion.AndroidRobolectricTestDeprecationLevel;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.DynamicMode;
import com.google.devtools.build.lib.rules.cpp.CppOptions.DynamicModeConverter;
import com.google.devtools.build.lib.rules.cpp.CppOptions.LibcTopLabelConverter;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import java.util.List;
import javax.annotation.Nullable;

/** Configuration fragment for Android rules. */
@AutoCodec
@SkylarkModule(
  name = "android",
  doc = "A configuration fragment for Android.",
  category = SkylarkModuleCategory.CONFIGURATION_FRAGMENT
)
@Immutable
public class AndroidConfiguration extends BuildConfiguration.Fragment {
  /**
   * Converter for {@link
   * com.google.devtools.build.lib.rules.android.AndroidConfiguration.ConfigurationDistinguisher}
   */
  public static final class ConfigurationDistinguisherConverter
      extends EnumConverter<ConfigurationDistinguisher> {
    public ConfigurationDistinguisherConverter() {
      super(ConfigurationDistinguisher.class, "Android configuration distinguisher");
    }
  }

  /** Converter for {@link ApkSigningMethod}. */
  public static final class ApkSigningMethodConverter extends EnumConverter<ApkSigningMethod> {
    public ApkSigningMethodConverter() {
      super(ApkSigningMethod.class, "apk signing method");
    }
  }

  /** Converter for {@link AndroidManifestMerger} */
  public static final class AndroidManifestMergerConverter
      extends EnumConverter<AndroidManifestMerger> {
    public AndroidManifestMergerConverter() {
      super(AndroidManifestMerger.class, "android manifest merger");
    }
  }

  /** Converter for {@link AndroidAaptVersion} */
  public static final class AndroidAaptConverter extends EnumConverter<AndroidAaptVersion> {
    public AndroidAaptConverter() {
      super(AndroidAaptVersion.class, "android androidAaptVersion");
    }
  }

  /** Converter for {@link AndroidRobolectricTestDeprecationLevel} */
  public static final class AndroidRobolectricTestDeprecationLevelConverter
      extends EnumConverter<AndroidRobolectricTestDeprecationLevel> {
    public AndroidRobolectricTestDeprecationLevelConverter() {
      super(AndroidRobolectricTestDeprecationLevel.class, "android robolectric deprecation level");
    }
  }

  /**
   * Value used to avoid multiple configurations from conflicting.
   *
   * <p>This is set to {@code ANDROID} in Android configurations and to {@code MAIN} otherwise. This
   * influences the output directory name: if it didn't, an Android and a non-Android configuration
   * would conflict if they had the same toolchain identifier.
   *
   * <p>Note that this is not just a theoretical concern: even if {@code --crosstool_top} and {@code
   * --android_crosstool_top} point to different labels, they may end up being redirected to the
   * same thing, and this is exactly what happens on OSX X.
   */
  public enum ConfigurationDistinguisher {
    MAIN(null),
    ANDROID("android");

    private final String suffix;

    ConfigurationDistinguisher(String suffix) {
      this.suffix = suffix;
    }
  }

  /**
   * Which APK signing method to use with the debug key for rules that build APKs.
   *
   * <ul>
   *   <li>V1 uses the apksigner attribute from the android_sdk and signs the APK as a JAR.
   *   <li>V2 uses the apksigner attribute from the android_sdk and signs the APK according to the
   *       APK Signing Schema V2 that is only supported on Android N and later.
   * </ul>
   */
  public enum ApkSigningMethod {
    V1(true, false),
    V2(false, true),
    V1_V2(true, true);

    private final boolean signV1;
    private final boolean signV2;

    ApkSigningMethod(boolean signV1, boolean signV2) {
      this.signV1 = signV1;
      this.signV2 = signV2;
    }

    /** Whether to JAR sign the APK with the apksigner tool. */
    public boolean signV1() {
      return signV1;
    }

    /** Whether to sign the APK with the apksigner tool with APK Signature Schema V2. */
    public boolean signV2() {
      return signV2;
    }
  }

  /** Types of android manifest mergers. */
  public enum AndroidManifestMerger {
    LEGACY,
    ANDROID;

    public static List<String> getAttributeValues() {
      return ImmutableList.of(
          LEGACY.name().toLowerCase(), ANDROID.name().toLowerCase(), getRuleAttributeDefault());
    }

    public static String getRuleAttributeDefault() {
      return "auto";
    }

    public static AndroidManifestMerger fromString(String value) {
      for (AndroidManifestMerger merger : AndroidManifestMerger.values()) {
        if (merger.name().equalsIgnoreCase(value)) {
          return merger;
        }
      }
      return null;
    }
  }

  /** Types of android manifest mergers. */
  public enum AndroidAaptVersion {
    AAPT,
    AAPT2,
    AUTO;

    public static List<String> getAttributeValues() {
      return ImmutableList.of(
          AAPT.name().toLowerCase(), AAPT2.name().toLowerCase(), getRuleAttributeDefault());
    }

    public static String getRuleAttributeDefault() {
      return AUTO.name().toLowerCase();
    }

    public static AndroidAaptVersion fromString(String value) {
      for (AndroidAaptVersion version : AndroidAaptVersion.values()) {
        if (version.name().equalsIgnoreCase(value)) {
          return version;
        }
      }
      return null;
    }

    /** android_robolectric_test deprecation levels */
    public enum AndroidRobolectricTestDeprecationLevel {
      OFF,
      WARNING,
      DEPRECATED;

      public static List<String> getAttributeValues() {
        return ImmutableList.of(
            OFF.name().toLowerCase(),
            WARNING.name().toLowerCase(),
            DEPRECATED.name().toLowerCase());
      }

      public static AndroidRobolectricTestDeprecationLevel fromString(String value) {
        for (AndroidRobolectricTestDeprecationLevel level :
            AndroidRobolectricTestDeprecationLevel.values()) {
          if (level.name().equals(value)) {
            return level;
          }
        }
        return null;
      }
    }

    // TODO(corysmith): Move to ApplicationManifest when no longer needed as a public function.
    @Nullable
    public static AndroidAaptVersion chooseTargetAaptVersion(RuleContext ruleContext)
        throws RuleErrorException {
      if (ruleContext.isLegalFragment(AndroidConfiguration.class)) {
        boolean hasAapt2 = AndroidSdkProvider.fromRuleContext(ruleContext).getAapt2() != null;

        if (ruleContext.getRule().isAttrDefined("aapt_version", STRING)) {
          // On rules that can choose a version, test attribute then flag choose the aapt version
          // target.
          return chooseTargetAaptVersion(
              ruleContext,
              ruleContext.getFragment(AndroidConfiguration.class),
              ruleContext.attributes().get("aapt_version", STRING));
        } else {
          // On rules can't choose, assume aapt2 if aapt2 is present in the sdk.
          return hasAapt2 ? AAPT2 : AAPT;
        }
      }
      return null;
    }

    @Nullable
    public static AndroidAaptVersion chooseTargetAaptVersion(
        RuleContext ruleContext, AndroidConfiguration androidConfig, @Nullable String versionString)
        throws RuleErrorException {

      boolean hasAapt2 = AndroidSdkProvider.fromRuleContext(ruleContext).getAapt2() != null;
      AndroidAaptVersion flag = androidConfig.getAndroidAaptVersion();

      AndroidAaptVersion version = fromString(versionString);
      // version is null if the value is "auto"
      version = version == AndroidAaptVersion.AUTO ? flag : version;

      if (version == AAPT2 && !hasAapt2) {
        ruleContext.throwWithRuleError(
            "aapt2 processing requested but not available on the android_sdk");
        return null;
      }
      return version == AndroidAaptVersion.AUTO ? AAPT : version;
    }
  }

  /** Android configuration options. */
  @AutoCodec(strategy = AutoCodec.Strategy.PUBLIC_FIELDS)
  public static class Options extends FragmentOptions {
    @Option(
      name = "Android configuration distinguisher",
      defaultValue = "MAIN",
      converter = ConfigurationDistinguisherConverter.class,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION,
      metadataTags = {OptionMetadataTag.INTERNAL}
    )
    public ConfigurationDistinguisher configurationDistinguisher;

    // For deploying incremental installation of native libraries. Do not use on the command line.
    // The idea is that once this option works, we'll flip the default value in a config file, then
    // once it is proven that it works, remove it from Bazel and said config file.
    @Option(
      name = "android_incremental_native_libs",
      defaultValue = "false",
      // mobile-install v1 is going away, and this flag does not apply to v2
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.EXECUTION,
        OptionEffectTag.LOADING_AND_ANALYSIS,
      },
      metadataTags = OptionMetadataTag.EXPERIMENTAL
    )
    public boolean incrementalNativeLibs;

    @Option(
      name = "android_crosstool_top",
      defaultValue = "//external:android/crosstool",
      converter = EmptyToNullLabelConverter.class,
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.CHANGES_INPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
      },
      help = "The location of the C++ compiler used for Android builds."
    )
    public Label androidCrosstoolTop;

    @Option(
      name = "android_cpu",
      defaultValue = "armeabi-v7a",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
      },
      help = "The Android target CPU."
    )
    public String cpu;

    @Option(
      name = "android_compiler",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
      },
      help = "The Android target compiler."
    )
    public String cppCompiler;

    @Option(
      name = "android_grte_top",
      defaultValue = "null",
      converter = LibcTopLabelConverter.class,
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {
        OptionEffectTag.CHANGES_INPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
      },
      help = "The Android target grte_top."
    )
    public Label androidLibcTopLabel;

    @Option(
      name = "android_dynamic_mode",
      defaultValue = "off",
      converter = DynamicModeConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS,
      },
      help =
          "Determines whether C++ deps of Android rules will be linked dynamically when a "
              + "cc_binary does not explicitly create a shared library. "
              + "'default' means blaze will choose whether to link dynamically.  "
              + "'fully' means all libraries will be linked dynamically. "
              + "'off' means that all libraries will be linked in mostly static mode."
    )
    public DynamicMode dynamicMode;

    // Label of filegroup combining all Android tools used as implicit dependencies of
    // android_* rules
    @Option(
      name = "android_sdk",
      defaultValue = "@bazel_tools//tools/android:sdk",
      converter = LabelConverter.class,
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {
        OptionEffectTag.CHANGES_INPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
      },
      help = "Specifies Android SDK/platform that is used to build Android applications."
    )
    public Label sdk;

    // TODO(bazel-team): Maybe merge this with --android_cpu above.
    @Option(
      name = "fat_apk_cpu",
      converter = Converters.CommaSeparatedOptionListConverter.class,
      defaultValue = "armeabi-v7a",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
      },
      help =
          "Setting this option enables fat APKs, which contain native binaries for all "
              + "specified target architectures, e.g., --fat_apk_cpu=x86,armeabi-v7a. If this "
              + "flag is specified, then --android_cpu is ignored for dependencies of "
              + "android_binary rules."
    )
    public List<String> fatApkCpus;

    // For desugaring lambdas when compiling Java 8 sources. Do not use on the command line.
    // The idea is that once this option works, we'll flip the default value in a config file, then
    // once it is proven that it works, remove it from Bazel and said config file.
    @Option(
      name = "desugar_for_android",
      oldName = "experimental_desugar_for_android",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
      },
      help = "Whether to desugar Java 8 bytecode before dexing."
    )
    public boolean desugarJava8;

    @Option(
      name = "experimental_desugar_java8_libs",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
      },
      metadataTags = OptionMetadataTag.EXPERIMENTAL,
      help = "Whether to include supported Java 8 libraries in apps for legacy devices."
    )
    public boolean desugarJava8Libs;

    // This flag is intended to be flipped globally.
    @Option(
      name = "experimental_check_desugar_deps",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {
        OptionEffectTag.EAGERNESS_TO_EXIT,
        OptionEffectTag.LOADING_AND_ANALYSIS,
      },
      metadataTags = OptionMetadataTag.EXPERIMENTAL,
      help = "Whether to double-check correct desugaring at Android binary level."
    )
    public boolean checkDesugarDeps;

    @Option(
      name = "incremental_dexing",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
      },
      help = "Does most of the work for dexing separately for each Jar file."
    )
    public boolean incrementalDexing;

    @Option(
      name = "experimental_incremental_dexing_after_proguard",
      defaultValue = "1",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.LOSES_INCREMENTAL_STATE},
      help =
          "Whether to use incremental dexing tools when building proguarded Android binaries.  "
              + "Values > 0 turn the feature on, values > 1 run that many dexbuilder shards."
    )
    public int incrementalDexingShardsAfterProguard;

    /** Whether to use a separate tool to shard classes before merging them into final dex files. */
    @Option(
      name = "experimental_use_dex_splitter_for_incremental_dexing",
      defaultValue = "false",
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      help = "Do not use."
    )
    public boolean incrementalDexingUseDexSharder;

    @Option(
      name = "experimental_incremental_dexing_after_proguard_by_default",
      defaultValue = "false",
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      help =
          "Whether to use incremental dexing for proguarded Android binaries by default.  "
              + "Use incremental_dexing attribute to override default for a particular "
              + "android_binary."
    )
    public boolean incrementalDexingAfterProguardByDefault;

    // TODO(b/31711689): Remove this flag when this optimization is proven to work globally.
    @Option(
      name = "experimental_android_assume_minsdkversion",
      defaultValue = "false",
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.ACTION_COMMAND_LINES,
        OptionEffectTag.LOADING_AND_ANALYSIS,
      },
      help =
          "When enabled, the minSdkVersion is parsed from the merged AndroidManifest and used to "
              + "instruct Proguard on valid Android build versions."
    )
    public boolean assumeMinSdkVersion;

    @Option(
      name = "experimental_android_use_parallel_dex2oat",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.TESTING,
      effectTags = {
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS
      },
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help = "Use dex2oat in parallel to possibly speed up android_test."
    )
    public boolean useParallelDex2Oat;

    // Do not use on the command line.
    // This flag is intended to be updated as we add supported flags to the incremental dexing tools
    @Option(
      name = "non_incremental_per_target_dexopts",
      converter = Converters.CommaSeparatedOptionListConverter.class,
      defaultValue = "--positions",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
      },
      help =
          "dx flags that that prevent incremental dexing for binary targets that list any of "
              + "the flags listed here in their 'dexopts' attribute, which are ignored with "
              + "incremental dexing (superseding --dexopts_supported_in_incremental_dexing).  "
              + "Defaults to --positions for safety but can in general be used "
              + "to make sure the listed dx flags are honored, with additional build latency.  "
              + "Please notify us if you find yourself needing this flag."
    )
    public List<String> nonIncrementalPerTargetDexopts;

    // Do not use on the command line.
    // This flag is intended to be updated as we add supported flags to the incremental dexing tools
    @Option(
      name = "dexopts_supported_in_incremental_dexing",
      converter = Converters.CommaSeparatedOptionListConverter.class,
      defaultValue = "--no-optimize,--no-locals",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.ACTION_COMMAND_LINES,
        OptionEffectTag.LOADING_AND_ANALYSIS,
      },
      help = "dx flags supported when converting Jars to dex archives incrementally."
    )
    public List<String> dexoptsSupportedInIncrementalDexing;

    // Do not use on the command line.
    // This flag is intended to be updated as we add supported flags to the incremental dexing tools
    @Option(
      name = "dexopts_supported_in_dexmerger",
      converter = Converters.CommaSeparatedOptionListConverter.class,
      defaultValue = "--minimal-main-dex,--set-max-idx-number",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.ACTION_COMMAND_LINES,
        OptionEffectTag.LOADING_AND_ANALYSIS,
      },
      help = "dx flags supported in tool that merges dex archives into final classes.dex files."
    )
    public List<String> dexoptsSupportedInDexMerger;

    @Option(
      name = "use_workers_with_dexbuilder",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.EXECUTION},
      help = "Whether dexbuilder supports being run in local worker mode."
    )
    public boolean useWorkersWithDexbuilder;

    @Option(
      name = "experimental_android_rewrite_dexes_with_rex",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
      },
      metadataTags = OptionMetadataTag.EXPERIMENTAL,
      help = "use rex tool to rewrite dex files"
    )
    public boolean useRexToCompressDexFiles;

    @Option(
      name = "experimental_allow_android_library_deps_without_srcs",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {
        OptionEffectTag.EAGERNESS_TO_EXIT,
        OptionEffectTag.LOADING_AND_ANALYSIS,
      },
      help =
          "Flag to help transition from allowing to disallowing srcs-less android_library"
              + " rules with deps. The depot needs to be cleaned up to roll this out by default."
    )
    public boolean allowAndroidLibraryDepsWithoutSrcs;

    @Option(
      name = "experimental_android_resource_shrinking",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS,
      },
      help = "Enables resource shrinking for android_binary APKs that use ProGuard."
    )
    public boolean useExperimentalAndroidResourceShrinking;

    @Option(
      name = "android_resource_shrinking",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS,
      },
      help = "Enables resource shrinking for android_binary APKs that use ProGuard."
    )
    public boolean useAndroidResourceShrinking;

    @Option(
      name = "experimental_android_resource_cycle_shrinking",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS,
      },
      metadataTags = OptionMetadataTag.EXPERIMENTAL,
      help =
          "Enables more shrinking of code and resources by instructing AAPT2 "
              + "to emit conditional Proguard keep rules."
    )
    public boolean useAndroidResourceCycleShrinking;

    @Option(
      name = "android_manifest_merger",
      defaultValue = "android",
      converter = AndroidManifestMergerConverter.class,
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
      },
      help =
          "Selects the manifest merger to use for android_binary rules. Flag to help the"
              + "transition to the Android manifest merger from the legacy merger."
    )
    public AndroidManifestMerger manifestMerger;

    @Option(
      name = "android_aapt",
      defaultValue = "aapt",
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
      },
      converter = AndroidAaptConverter.class,
      help =
          "Selects the version of androidAaptVersion to use for android_binary rules."
              + "Flag to help the test and transition to aapt2."
    )
    public AndroidAaptVersion androidAaptVersion;

    @Option(
      name = "apk_signing_method",
      converter = ApkSigningMethodConverter.class,
      defaultValue = "v1_v2",
      documentationCategory = OptionDocumentationCategory.SIGNING,
      effectTags = {
        OptionEffectTag.ACTION_COMMAND_LINES,
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS,
      },
      help = "Implementation to use to sign APKs"
    )
    public ApkSigningMethod apkSigningMethod;

    @Option(
      name = "use_singlejar_apkbuilder",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
      effectTags = OptionEffectTag.LOADING_AND_ANALYSIS,
      help = "Build Android APKs with SingleJar."
    )
    public boolean useSingleJarApkBuilder;

    @Option(
      name = "experimental_android_compress_java_resources",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS,
      },
      metadataTags = OptionMetadataTag.EXPERIMENTAL,
      help = "Compress Java resources in APKs"
    )
    public boolean compressJavaResources;

    @Option(
      name = "experimental_android_library_exports_manifest_default",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
      },
      help = "The default value of the exports_manifest attribute on android_library."
    )
    public boolean exportsManifestDefault;

    @Option(
      name = "experimental_android_aapt2_robolectric",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.LOADING_AND_ANALYSIS,
      },
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help = "If true, robolectric resources will be packaged using aapt2 if available."
    )
    public boolean useAapt2ForRobolectric;

    @Option(
      name = "experimental_android_throw_on_resource_conflict",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {
        OptionEffectTag.EAGERNESS_TO_EXIT,
        OptionEffectTag.LOADING_AND_ANALYSIS,
      },
      help = "If passed, resource merge conflicts will be treated as errors instead of warnings"
    )
    public boolean throwOnResourceConflict;

    @Option(
      name = "experimental_skip_parsing_action",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Skips resource parsing action for library targets"
              + " and uses the output of the compile action instead for resource merging."
    )
    public boolean skipParsingAction;

    @Option(
      name = "android_fixed_resource_neverlinking",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      help =
          "If true, resources will properly not get propagated through neverlinked libraries."
              + " Otherwise, the old behavior of propagating those resources if no resource-related"
              + " attributes are specified in the neverlink library will be preserved."
    )
    public boolean fixedResourceNeverlinking;

    @Option(
      name = "android_robolectric_test_deprecation_level",
      defaultValue = "off",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.BUILD_FILE_SEMANTICS},
      converter = AndroidRobolectricTestDeprecationLevelConverter.class,
      help =
          "Determine the deprecation level of android_robolectric_test. Can be 'off', "
              + "'warning', or 'deprecated'."
    )
    public AndroidRobolectricTestDeprecationLevel robolectricTestDeprecationLevel;

    @Option(
        name = "android_decouple_data_processing",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {
          OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION,
          OptionEffectTag.ACTION_COMMAND_LINES
        },
        help =
            "If true, Android data (assets, resources, and manifests) will be processed seperately "
                + "when possible. Otherwise, they will all be processed together.")
    public boolean decoupleDataProcessing;

    // TODO(cushon): make this the default, and delete it
    @Option(
      name = "experimental_android_enforce_strict_deps_for_binaries_under_test",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      help = "If enabled, strict dependencies are enforced for android_test.binary_under_test"
    )
    public boolean enforceStrictDepsForBinariesUnderTest;

    @Option(
      name = "android_migration_tag_check",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.EAGERNESS_TO_EXIT,
      },
      help = "If enabled, strict usage of the Skylark migration tag is enabled for android rules."
    )
    public boolean checkForMigrationTag;

    // TODO(eaftan): enable this by default and delete it
    @Option(
      name = "experimental_one_version_enforcement_use_transitive_jars_for_binary_under_test",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION,
        OptionEffectTag.ACTION_COMMAND_LINES
      },
      help =
          "If enabled, one version enforcement for android_test uses the binary_under_test's "
              + "transitive classpath, otherwise it uses the deploy jar"
    )
    public boolean oneVersionEnforcementUseTransitiveJarsForBinaryUnderTest;

    @Override
    public FragmentOptions getHost() {
      Options host = (Options) super.getHost();
      host.androidCrosstoolTop = androidCrosstoolTop;
      host.sdk = sdk;
      host.fatApkCpus = ImmutableList.of(); // Fat APK archs don't apply to the host.

      host.desugarJava8 = desugarJava8;
      host.desugarJava8Libs = desugarJava8Libs;
      host.checkDesugarDeps = checkDesugarDeps;
      host.incrementalDexing = incrementalDexing;
      host.incrementalDexingShardsAfterProguard = incrementalDexingShardsAfterProguard;
      host.incrementalDexingUseDexSharder = incrementalDexingUseDexSharder;
      host.incrementalDexingAfterProguardByDefault = incrementalDexingAfterProguardByDefault;
      host.assumeMinSdkVersion = assumeMinSdkVersion;
      host.nonIncrementalPerTargetDexopts = nonIncrementalPerTargetDexopts;
      host.dexoptsSupportedInIncrementalDexing = dexoptsSupportedInIncrementalDexing;
      host.dexoptsSupportedInDexMerger = dexoptsSupportedInDexMerger;
      host.useWorkersWithDexbuilder = useWorkersWithDexbuilder;
      host.manifestMerger = manifestMerger;
      host.androidAaptVersion = androidAaptVersion;
      host.allowAndroidLibraryDepsWithoutSrcs = allowAndroidLibraryDepsWithoutSrcs;
      host.enforceStrictDepsForBinariesUnderTest = enforceStrictDepsForBinariesUnderTest;
      host.oneVersionEnforcementUseTransitiveJarsForBinaryUnderTest =
          oneVersionEnforcementUseTransitiveJarsForBinaryUnderTest;
      return host;
    }
  }

  /** Configuration loader for the Android fragment. */
  public static class Loader implements ConfigurationFragmentFactory {
    @Override
    public Fragment create(BuildOptions buildOptions)
        throws InvalidConfigurationException, InterruptedException {
      return new AndroidConfiguration(buildOptions.get(Options.class));
    }

    @Override
    public Class<? extends Fragment> creates() {
      return AndroidConfiguration.class;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.of(Options.class);
    }
  }

  private final Label sdk;
  private final String cpu;
  private final boolean useIncrementalNativeLibs;
  private final ConfigurationDistinguisher configurationDistinguisher;
  private final boolean incrementalDexing;
  private final int incrementalDexingShardsAfterProguard;
  private final boolean incrementalDexingUseDexSharder;
  private final boolean incrementalDexingAfterProguardByDefault;
  private final boolean assumeMinSdkVersion;
  private final ImmutableList<String> dexoptsSupportedInIncrementalDexing;
  private final ImmutableList<String> targetDexoptsThatPreventIncrementalDexing;
  private final ImmutableList<String> dexoptsSupportedInDexMerger;
  private final boolean useWorkersWithDexbuilder;
  private final boolean desugarJava8;
  private final boolean desugarJava8Libs;
  private final boolean checkDesugarDeps;
  private final boolean useRexToCompressDexFiles;
  private final boolean allowAndroidLibraryDepsWithoutSrcs;
  private final boolean useAndroidResourceShrinking;
  private final boolean useAndroidResourceCycleShrinking;
  private final AndroidManifestMerger manifestMerger;
  private final ApkSigningMethod apkSigningMethod;
  private final boolean useSingleJarApkBuilder;
  private final boolean compressJavaResources;
  private final boolean exportsManifestDefault;
  private final AndroidAaptVersion androidAaptVersion;
  private final boolean useAapt2ForRobolectric;
  private final boolean throwOnResourceConflict;
  private final boolean useParallelDex2Oat;
  private final boolean skipParsingAction;
  private final boolean fixedResourceNeverlinking;
  private final AndroidRobolectricTestDeprecationLevel robolectricTestDeprecationLevel;
  private final boolean decoupleDataProcessing;
  private final boolean enforceStrictDepsForBinariesUnderTest;
  private final boolean checkForMigrationTag;
  private final boolean oneVersionEnforcementUseTransitiveJarsForBinaryUnderTest;

  AndroidConfiguration(Options options) throws InvalidConfigurationException {
    this.sdk = options.sdk;
    this.useIncrementalNativeLibs = options.incrementalNativeLibs;
    this.cpu = options.cpu;
    this.configurationDistinguisher = options.configurationDistinguisher;
    this.incrementalDexing = options.incrementalDexing;
    this.incrementalDexingShardsAfterProguard = options.incrementalDexingShardsAfterProguard;
    this.incrementalDexingUseDexSharder = options.incrementalDexingUseDexSharder;
    this.incrementalDexingAfterProguardByDefault = options.incrementalDexingAfterProguardByDefault;
    this.assumeMinSdkVersion = options.assumeMinSdkVersion;
    this.dexoptsSupportedInIncrementalDexing =
        ImmutableList.copyOf(options.dexoptsSupportedInIncrementalDexing);
    this.targetDexoptsThatPreventIncrementalDexing =
        ImmutableList.copyOf(options.nonIncrementalPerTargetDexopts);
    this.dexoptsSupportedInDexMerger = ImmutableList.copyOf(options.dexoptsSupportedInDexMerger);
    this.useWorkersWithDexbuilder = options.useWorkersWithDexbuilder;
    this.desugarJava8 = options.desugarJava8;
    this.desugarJava8Libs = options.desugarJava8Libs;
    this.checkDesugarDeps = options.checkDesugarDeps;
    this.allowAndroidLibraryDepsWithoutSrcs = options.allowAndroidLibraryDepsWithoutSrcs;
    this.useAndroidResourceShrinking =
        options.useAndroidResourceShrinking || options.useExperimentalAndroidResourceShrinking;
    this.useAndroidResourceCycleShrinking = options.useAndroidResourceCycleShrinking;
    this.manifestMerger = options.manifestMerger;
    this.apkSigningMethod = options.apkSigningMethod;
    this.useSingleJarApkBuilder = options.useSingleJarApkBuilder;
    this.useRexToCompressDexFiles = options.useRexToCompressDexFiles;
    this.compressJavaResources = options.compressJavaResources;
    this.exportsManifestDefault = options.exportsManifestDefault;
    this.androidAaptVersion = options.androidAaptVersion;
    this.useAapt2ForRobolectric = options.useAapt2ForRobolectric;
    this.throwOnResourceConflict = options.throwOnResourceConflict;
    this.useParallelDex2Oat = options.useParallelDex2Oat;
    this.skipParsingAction = options.skipParsingAction;
    this.fixedResourceNeverlinking = options.fixedResourceNeverlinking;
    this.robolectricTestDeprecationLevel = options.robolectricTestDeprecationLevel;
    this.decoupleDataProcessing = options.decoupleDataProcessing;
    this.enforceStrictDepsForBinariesUnderTest = options.enforceStrictDepsForBinariesUnderTest;
    this.checkForMigrationTag = options.checkForMigrationTag;
    this.oneVersionEnforcementUseTransitiveJarsForBinaryUnderTest =
        options.oneVersionEnforcementUseTransitiveJarsForBinaryUnderTest;

    if (incrementalDexingShardsAfterProguard < 0) {
      throw new InvalidConfigurationException(
          "--experimental_incremental_dexing_after_proguard must be a positive number");
    }
    if (incrementalDexingAfterProguardByDefault && incrementalDexingShardsAfterProguard == 0) {
      throw new InvalidConfigurationException(
          "--experimental_incremental_dexing_after_proguard_by_default requires "
              + "--experimental_incremental_dexing_after_proguard to be at least 1");
    }
    if (desugarJava8Libs && !desugarJava8) {
      throw new InvalidConfigurationException(
          "Java 8 library support requires --desugar_java8 to be enabled.");
    }
  }

  @AutoCodec.Instantiator
  AndroidConfiguration(
      Label sdk,
      String cpu,
      boolean useIncrementalNativeLibs,
      ConfigurationDistinguisher configurationDistinguisher,
      boolean incrementalDexing,
      int incrementalDexingShardsAfterProguard,
      boolean incrementalDexingUseDexSharder,
      boolean incrementalDexingAfterProguardByDefault,
      boolean assumeMinSdkVersion,
      ImmutableList<String> dexoptsSupportedInIncrementalDexing,
      ImmutableList<String> targetDexoptsThatPreventIncrementalDexing,
      ImmutableList<String> dexoptsSupportedInDexMerger,
      boolean useWorkersWithDexbuilder,
      boolean desugarJava8,
      boolean desugarJava8Libs,
      boolean checkDesugarDeps,
      boolean useRexToCompressDexFiles,
      boolean allowAndroidLibraryDepsWithoutSrcs,
      boolean useAndroidResourceShrinking,
      boolean useAndroidResourceCycleShrinking,
      AndroidManifestMerger manifestMerger,
      ApkSigningMethod apkSigningMethod,
      boolean useSingleJarApkBuilder,
      boolean compressJavaResources,
      boolean exportsManifestDefault,
      AndroidAaptVersion androidAaptVersion,
      boolean useAapt2ForRobolectric,
      boolean throwOnResourceConflict,
      boolean useParallelDex2Oat,
      boolean skipParsingAction,
      boolean fixedResourceNeverlinking,
      AndroidRobolectricTestDeprecationLevel robolectricTestDeprecationLevel,
      boolean decoupleDataProcessing,
      boolean enforceStrictDepsForBinariesUnderTest,
      boolean checkForMigrationTag,
      boolean oneVersionEnforcementUseTransitiveJarsForBinaryUnderTest) {
    this.sdk = sdk;
    this.cpu = cpu;
    this.useIncrementalNativeLibs = useIncrementalNativeLibs;
    this.configurationDistinguisher = configurationDistinguisher;
    this.incrementalDexing = incrementalDexing;
    this.incrementalDexingShardsAfterProguard = incrementalDexingShardsAfterProguard;
    this.incrementalDexingUseDexSharder = incrementalDexingUseDexSharder;
    this.incrementalDexingAfterProguardByDefault = incrementalDexingAfterProguardByDefault;
    this.assumeMinSdkVersion = assumeMinSdkVersion;
    this.dexoptsSupportedInIncrementalDexing = dexoptsSupportedInIncrementalDexing;
    this.targetDexoptsThatPreventIncrementalDexing = targetDexoptsThatPreventIncrementalDexing;
    this.dexoptsSupportedInDexMerger = dexoptsSupportedInDexMerger;
    this.useWorkersWithDexbuilder = useWorkersWithDexbuilder;
    this.desugarJava8 = desugarJava8;
    this.desugarJava8Libs = desugarJava8Libs;
    this.checkDesugarDeps = checkDesugarDeps;
    this.useRexToCompressDexFiles = useRexToCompressDexFiles;
    this.allowAndroidLibraryDepsWithoutSrcs = allowAndroidLibraryDepsWithoutSrcs;
    this.useAndroidResourceShrinking = useAndroidResourceShrinking;
    this.useAndroidResourceCycleShrinking = useAndroidResourceCycleShrinking;
    this.manifestMerger = manifestMerger;
    this.apkSigningMethod = apkSigningMethod;
    this.useSingleJarApkBuilder = useSingleJarApkBuilder;
    this.compressJavaResources = compressJavaResources;
    this.exportsManifestDefault = exportsManifestDefault;
    this.androidAaptVersion = androidAaptVersion;
    this.useAapt2ForRobolectric = useAapt2ForRobolectric;
    this.throwOnResourceConflict = throwOnResourceConflict;
    this.useParallelDex2Oat = useParallelDex2Oat;
    this.skipParsingAction = skipParsingAction;
    this.fixedResourceNeverlinking = fixedResourceNeverlinking;
    this.robolectricTestDeprecationLevel = robolectricTestDeprecationLevel;
    this.decoupleDataProcessing = decoupleDataProcessing;
    this.enforceStrictDepsForBinariesUnderTest = enforceStrictDepsForBinariesUnderTest;
    this.checkForMigrationTag = checkForMigrationTag;
    this.oneVersionEnforcementUseTransitiveJarsForBinaryUnderTest =
        oneVersionEnforcementUseTransitiveJarsForBinaryUnderTest;
  }

  public String getCpu() {
    return cpu;
  }

  @SkylarkConfigurationField(
    name = "android_sdk_label",
    doc = "Returns the target denoted by the value of the --android_sdk flag",
    defaultLabel = AndroidRuleClasses.DEFAULT_SDK,
    defaultInToolRepository = true
  )
  public Label getSdk() {
    return sdk;
  }

  public boolean useIncrementalNativeLibs() {
    return useIncrementalNativeLibs;
  }

  /** Returns whether to use incremental dexing. */
  public boolean useIncrementalDexing() {
    return incrementalDexing;
  }

  /** Returns whether to process proguarded Android binaries with incremental dexing tools. */
  public int incrementalDexingShardsAfterProguard() {
    return incrementalDexingShardsAfterProguard;
  }

  /** Whether to use a separate tool to shard classes before merging them into final dex files. */
  public boolean incrementalDexingUseDexSharder() {
    return incrementalDexingUseDexSharder;
  }

  /** Whether to use incremental dexing to build proguarded binaries by default. */
  public boolean incrementalDexingAfterProguardByDefault() {
    return incrementalDexingAfterProguardByDefault;
  }

  /**
   * Returns true if an -assumevalues should be generated for Proguard based on the minSdkVersion of
   * the merged AndroidManifest.
   */
  public boolean assumeMinSdkVersion() {
    return assumeMinSdkVersion;
  }

  /** dx flags supported in incremental dexing actions. */
  public ImmutableList<String> getDexoptsSupportedInIncrementalDexing() {
    return dexoptsSupportedInIncrementalDexing;
  }

  /** dx flags supported in dexmerger actions. */
  public ImmutableList<String> getDexoptsSupportedInDexMerger() {
    return dexoptsSupportedInDexMerger;
  }

  /**
   * Incremental dexing must not be used for binaries that list any of these flags in their {@code
   * dexopts} attribute.
   */
  public ImmutableList<String> getTargetDexoptsThatPreventIncrementalDexing() {
    return targetDexoptsThatPreventIncrementalDexing;
  }

  /** Whether to assume the dexbuilder tool supports local worker mode. */
  public boolean useWorkersWithDexbuilder() {
    return useWorkersWithDexbuilder;
  }

  public boolean desugarJava8() {
    return desugarJava8;
  }

  public boolean desugarJava8Libs() {
    return desugarJava8Libs;
  }

  public boolean checkDesugarDeps() {
    return checkDesugarDeps;
  }

  public boolean useRexToCompressDexFiles() {
    return useRexToCompressDexFiles;
  }

  public boolean allowSrcsLessAndroidLibraryDeps() {
    return allowAndroidLibraryDepsWithoutSrcs;
  }

  public boolean useAndroidResourceShrinking() {
    return useAndroidResourceShrinking;
  }

  public boolean useAndroidResourceCycleShrinking() {
    return useAndroidResourceCycleShrinking;
  }

  public AndroidAaptVersion getAndroidAaptVersion() {
    return androidAaptVersion;
  }

  public AndroidManifestMerger getManifestMerger() {
    return manifestMerger;
  }

  public ApkSigningMethod getApkSigningMethod() {
    return apkSigningMethod;
  }

  public boolean useSingleJarApkBuilder() {
    return useSingleJarApkBuilder;
  }

  public boolean useParallelDex2Oat() {
    return useParallelDex2Oat;
  }

  boolean compressJavaResources() {
    return compressJavaResources;
  }

  boolean getExportsManifestDefault() {
    return exportsManifestDefault;
  }

  public boolean useAapt2ForRobolectric() {
    return useAapt2ForRobolectric;
  }

  boolean throwOnResourceConflict() {
    return throwOnResourceConflict;
  }

  public boolean skipParsingAction() {
    return this.skipParsingAction;
  }

  public boolean fixedResourceNeverlinking() {
    return this.fixedResourceNeverlinking;
  }

  public AndroidRobolectricTestDeprecationLevel getRobolectricTestDeprecationLevel() {
    return robolectricTestDeprecationLevel;
  }

  public boolean decoupleDataProcessing() {
    return decoupleDataProcessing;
  }

  public boolean getEnforceStrictDepsForBinariesUnderTest() {
    return enforceStrictDepsForBinariesUnderTest;
  }

  public boolean checkForMigrationTag() {
    return checkForMigrationTag;
  }

  public boolean getOneVersionEnforcementUseTransitiveJarsForBinaryUnderTest() {
    return oneVersionEnforcementUseTransitiveJarsForBinaryUnderTest;
  }

  @Override
  public void addGlobalMakeVariables(ImmutableMap.Builder<String, String> globalMakeEnvBuilder) {
    globalMakeEnvBuilder.put("ANDROID_CPU", cpu);
  }

  @Override
  public String getOutputDirectoryName() {
    return configurationDistinguisher.suffix;
  }
}
