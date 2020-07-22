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


import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.Allowlist;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.EmptyToNullLabelConverter;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.LabelConverter;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.skylark.annotations.StarlarkConfigurationField;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.DynamicMode;
import com.google.devtools.build.lib.rules.cpp.CppOptions.DynamicModeConverter;
import com.google.devtools.build.lib.rules.cpp.CppOptions.LibcTopLabelConverter;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidConfigurationApi;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import java.util.List;
import javax.annotation.Nullable;

/** Configuration fragment for Android rules. */
@Immutable
public class AndroidConfiguration extends Fragment implements AndroidConfigurationApi {

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

  /** Converter for {@link ManifestMergerOrder} */
  public static final class ManifestMergerOrderConverter
      extends EnumConverter<ManifestMergerOrder> {
    public ManifestMergerOrderConverter() {
      super(ManifestMergerOrder.class, "android manifest merger order");
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
    ANDROID,
    FORCE_ANDROID;

    public static List<String> getAttributeValues() {
      return ImmutableList.of(
          LEGACY.name().toLowerCase(),
          ANDROID.name().toLowerCase(),
          FORCE_ANDROID.name().toLowerCase(),
          getRuleAttributeDefault());
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

  /** Orders for merging android manifests. */
  public enum ManifestMergerOrder {
    /** Manifests are sorted alphabetically by exec path. */
    ALPHABETICAL,
    /** Manifests are sorted alphabetically by configuration-relative path. */
    ALPHABETICAL_BY_CONFIGURATION,
    /** Library manifests come before the manifests of their dependencies. */
    DEPENDENCY;
  }

  /** Android configuration options. */
  public static class Options extends FragmentOptions {
    @Option(
        name = "Android configuration distinguisher",
        defaultValue = "MAIN",
        converter = ConfigurationDistinguisherConverter.class,
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION,
        metadataTags = {OptionMetadataTag.INTERNAL})
    public ConfigurationDistinguisher configurationDistinguisher;

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
        help = "The location of the C++ compiler used for Android builds.")
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
        help = "The Android target CPU.")
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
        help = "The Android target compiler.")
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
        help = "The Android target grte_top.")
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
                + "'default' means bazel will choose whether to link dynamically.  "
                + "'fully' means all libraries will be linked dynamically. "
                + "'off' means that all libraries will be linked in mostly static mode.")
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
        help = "Specifies Android SDK/platform that is used to build Android applications.")
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
                + "android_binary rules.")
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
        help = "Whether to desugar Java 8 bytecode before dexing.")
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
        help = "Whether to include supported Java 8 libraries in apps for legacy devices.")
    public boolean desugarJava8Libs;

    // This flag is intended to be flipped globally.
    @Option(
        name = "experimental_check_desugar_deps",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
        effectTags = {
          OptionEffectTag.EAGERNESS_TO_EXIT,
          OptionEffectTag.LOADING_AND_ANALYSIS,
        },
        metadataTags = OptionMetadataTag.EXPERIMENTAL,
        help = "Whether to double-check correct desugaring at Android binary level.")
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
        help = "Does most of the work for dexing separately for each Jar file.")
    public boolean incrementalDexing;

    @Option(
        name = "experimental_incremental_dexing_after_proguard",
        defaultValue = "1",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {
          OptionEffectTag.LOADING_AND_ANALYSIS,
          OptionEffectTag.LOSES_INCREMENTAL_STATE
        },
        help =
            "Whether to use incremental dexing tools when building proguarded Android binaries.  "
                + "Values > 0 turn the feature on, values > 1 run that many dexbuilder shards.")
    public int incrementalDexingShardsAfterProguard;

    /** Whether to use a separate tool to shard classes before merging them into final dex files. */
    @Option(
        name = "experimental_use_dex_splitter_for_incremental_dexing",
        defaultValue = "false",
        metadataTags = {OptionMetadataTag.EXPERIMENTAL},
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
        help = "Do not use.")
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
                + "android_binary.")
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
                + "instruct Proguard on valid Android build versions.")
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
        help = "Use dex2oat in parallel to possibly speed up android_test.")
    public boolean useParallelDex2Oat;

    @Option(
        name = "break_build_on_parallel_dex2oat_failure",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.TESTING,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
        metadataTags = {OptionMetadataTag.EXPERIMENTAL},
        help =
            "If true dex2oat action failures will cause the build to break "
                + "instead of executing dex2oat during test runtime.")
    public boolean breakBuildOnParallelDex2OatFailure;

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
                + "Please notify us if you find yourself needing this flag.")
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
        help = "dx flags supported when converting Jars to dex archives incrementally.")
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
        help = "dx flags supported in tool that merges dex archives into final classes.dex files.")
    public List<String> dexoptsSupportedInDexMerger;

    // Do not use on the command line.
    // This flag is intended to be updated as we add supported flags to the incremental dexing tools
    @Option(
        name = "dexopts_supported_in_dexsharder",
        converter = Converters.CommaSeparatedOptionListConverter.class,
        defaultValue = "--minimal-main-dex",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {
          OptionEffectTag.ACTION_COMMAND_LINES,
          OptionEffectTag.LOADING_AND_ANALYSIS,
        },
        help = "dx flags supported in tool that groups classes for inclusion in final .dex files.")
    public List<String> dexoptsSupportedInDexSharder;

    @Option(
        name = "use_workers_with_dexbuilder",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.EXECUTION},
        help = "Whether dexbuilder supports being run in local worker mode.")
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
        help = "use rex tool to rewrite dex files")
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
                + " rules with deps. The depot needs to be cleaned up to roll this out by default.")
    public boolean allowAndroidLibraryDepsWithoutSrcs;

    @Option(
        name = "experimental_android_resource_shrinking",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
        effectTags = {
          OptionEffectTag.AFFECTS_OUTPUTS,
          OptionEffectTag.LOADING_AND_ANALYSIS,
        },
        help = "Enables resource shrinking for android_binary APKs that use ProGuard.")
    public boolean useExperimentalAndroidResourceShrinking;

    @Option(
        name = "android_resource_shrinking",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
        effectTags = {
          OptionEffectTag.AFFECTS_OUTPUTS,
          OptionEffectTag.LOADING_AND_ANALYSIS,
        },
        help = "Enables resource shrinking for android_binary APKs that use ProGuard.")
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
                + "to emit conditional Proguard keep rules.")
    public boolean useAndroidResourceCycleShrinking;

    @Option(
        name = "experimental_android_resource_path_shortening",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {
          OptionEffectTag.AFFECTS_OUTPUTS,
          OptionEffectTag.LOADING_AND_ANALYSIS,
        },
        metadataTags = OptionMetadataTag.EXPERIMENTAL,
        help = "Enables shortening of resource file paths within android_binary APKs.")
    public boolean useAndroidResourcePathShortening;

    @Option(
        name = "experimental_android_resource_name_obfuscation",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {
          OptionEffectTag.AFFECTS_OUTPUTS,
          OptionEffectTag.LOADING_AND_ANALYSIS,
        },
        metadataTags = OptionMetadataTag.EXPERIMENTAL,
        help = "Enables obfuscation of resource names within android_binary APKs.")
    public boolean useAndroidResourceNameObfuscation;

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
                + "transition to the Android manifest merger from the legacy merger.")
    public AndroidManifestMerger manifestMerger;

    @Option(
        name = "android_manifest_merger_order",
        documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
        effectTags = {
          OptionEffectTag.ACTION_COMMAND_LINES,
          OptionEffectTag.EXECUTION,
        },
        defaultValue = "alphabetical",
        converter = ManifestMergerOrderConverter.class,
        help =
            "Sets the order of manifests passed to the manifest merger for Android binaries. "
                + "ALPHABETICAL means manifests are sorted by path relative to the execroot. "
                + "ALPHABETICAL_BY_CONFIGURATION means manifests are sorted by paths relative "
                + "to the configuration directory within the output directory. "
                + "DEPENDENCY means manifests are ordered with each library's manifest coming "
                + "before the manifests of its dependencies.")
    public ManifestMergerOrder manifestMergerOrder;

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
        help = "Implementation to use to sign APKs")
    public ApkSigningMethod apkSigningMethod;

    @Option(
        name = "use_singlejar_apkbuilder",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
        effectTags = OptionEffectTag.LOADING_AND_ANALYSIS,
        help = "Build Android APKs with SingleJar.")
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
        help = "Compress Java resources in APKs")
    public boolean compressJavaResources;

    @Option(
        name = "experimental_android_databinding_v2",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
        effectTags = {
          OptionEffectTag.AFFECTS_OUTPUTS,
          OptionEffectTag.LOADING_AND_ANALYSIS,
          OptionEffectTag.LOSES_INCREMENTAL_STATE,
        },
        metadataTags = OptionMetadataTag.EXPERIMENTAL,
        help = "Use android databinding v2")
    public boolean dataBindingV2;

    @Option(
        name = "android_databinding_use_v3_4_args",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
        effectTags = {
          OptionEffectTag.AFFECTS_OUTPUTS,
          OptionEffectTag.LOADING_AND_ANALYSIS,
          OptionEffectTag.LOSES_INCREMENTAL_STATE,
        },
        metadataTags = OptionMetadataTag.EXPERIMENTAL,
        help = "Use android databinding v2 with 3.4.0 argument")
    public boolean dataBindingUpdatedArgs;

    @Option(
        name = "experimental_android_library_exports_manifest_default",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {
          OptionEffectTag.AFFECTS_OUTPUTS,
          OptionEffectTag.LOADING_AND_ANALYSIS,
          OptionEffectTag.LOSES_INCREMENTAL_STATE,
        },
        help = "The default value of the exports_manifest attribute on android_library.")
    public boolean exportsManifestDefault;

    @Option(
        name = "experimental_omit_resources_info_provider_from_android_binary",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
        help =
            "Omit AndroidResourcesInfo provider from android_binary rules."
                + " Propagating resources out to other binaries is usually unintentional.")
    public boolean omitResourcesInfoProviderFromAndroidBinary;

    @Option(
        name = "android_fixed_resource_neverlinking",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
        help =
            "If true, resources will properly not get propagated through neverlinked libraries."
                + " Otherwise, the old behavior of propagating those resources if no"
                + " resource-related attributes are specified in the neverlink library"
                + " will be preserved.")
    public boolean fixedResourceNeverlinking;

    @Option(
        name = "android_migration_tag_check",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {
          OptionEffectTag.EAGERNESS_TO_EXIT,
        },
        help =
            "If enabled, strict usage of the Starlark migration tag is enabled for android rules. "
                + "Prefer using --incompatible_disable_native_android_rules.")
    public boolean checkForMigrationTag;

    @Option(
        name = "incompatible_disable_native_android_rules",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
        effectTags = {
          OptionEffectTag.EAGERNESS_TO_EXIT,
        },
        metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
        },
        help =
            "If enabled, direct usage of the native Android rules is disabled. Please use the"
                + " Starlark Android rules from https://github.com/bazelbuild/rules_android")
    public boolean disableNativeAndroidRules;

    @Option(
        name = "incompatible_use_toolchain_resolution",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
        metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
        },
        help =
            "Use toolchain resolution to select the Android SDK for android rules (Starlark and"
                + " native)")
    public boolean incompatibleUseToolchainResolution;

    @Option(
        name = "experimental_filter_r_jars_from_android_test",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {
          OptionEffectTag.CHANGES_INPUTS,
        },
        help = "If enabled, R Jars will be filtered from the test apk built by android_test.")
    public boolean filterRJarsFromAndroidTest;

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
                + "transitive classpath, otherwise it uses the deploy jar")
    public boolean oneVersionEnforcementUseTransitiveJarsForBinaryUnderTest;

    @Option(
        name = "persistent_android_resource_processor",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
        effectTags = {
          OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS,
          OptionEffectTag.EXECUTION,
        },
        help = "Enable the persistent Android resource processor by using workers.",
        expansion = {
          "--internal_persistent_busybox_tools",
          // This implementation uses unique workers for each tool in the busybox.
          "--strategy=AaptPackage=worker",
          "--strategy=AndroidResourceParser=worker",
          "--strategy=AndroidResourceValidator=worker",
          "--strategy=AndroidResourceCompiler=worker",
          "--strategy=RClassGenerator=worker",
          "--strategy=AndroidResourceLink=worker",
          "--strategy=AndroidAapt2=worker",
          "--strategy=AndroidAssetMerger=worker",
          "--strategy=AndroidResourceMerger=worker",
          "--strategy=AndroidCompiledResourceMerger=worker",
          "--strategy=ManifestMerger=worker",
          "--strategy=AndroidManifestMerger=worker",
          "--strategy=Aapt2Optimize=worker",
          "--strategy=AARGenerator=worker",
        })
    public Void persistentResourceProcessor;

    /**
     * We use this option to decide when to enable workers for busybox tools. This flag is also a
     * guard against enabling workers using nothing but --persistent_android_resource_processor.
     *
     * <p>Consequently, we use this option to decide between param files or regular command line
     * parameters. If we're not using workers or on Windows, there's no need to always use param
     * files for I/O performance reasons.
     */
    @Option(
        name = "internal_persistent_busybox_tools",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {
          OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS,
          OptionEffectTag.EXECUTION,
        },
        defaultValue = "false",
        help = "Tracking flag for when busybox workers are enabled.")
    public boolean persistentBusyboxTools;

    // TODO(b/142520065): Remove.
    @Option(
        name = "incompatible_prohibit_aapt1",
        documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
        effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.AFFECTS_OUTPUTS},
        metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
        },
        defaultValue = "true",
        help =
            "End support for aapt in Android rules. "
                + "To resolve issues when migrating your app to build with aapt2, see "
                + "https://developer.android.com/studio/command-line/aapt2#aapt2_changes")
    public boolean incompatibleProhibitAapt1;

    @Option(
        name = "experimental_remove_r_classes_from_instrumentation_test_jar",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {
          OptionEffectTag.CHANGES_INPUTS,
        },
        help =
            "If enabled and the test instruments an application, all the R classes from the test's "
                + "deploy jar will be removed.")
    public boolean removeRClassesFromInstrumentationTestJar;

    @Option(
        name = "experimental_always_filter_duplicate_classes_from_android_test",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {
          OptionEffectTag.CHANGES_INPUTS,
        },
        help =
            "If enabled and the android_test defines a binary_under_test, the class filterering "
                + "applied to the test's deploy jar will always filter duplicate classes based "
                + "solely on matching class and package name, ignoring hash values.")
    public boolean alwaysFilterDuplicateClassesFromAndroidTest;

    @Option(
        name = "experimental_filter_library_jar_with_program_jar",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
        effectTags = {OptionEffectTag.ACTION_COMMAND_LINES},
        help =
            "Filter the ProGuard ProgramJar to remove any classes also present in the LibraryJar.")
    public boolean filterLibraryJarWithProgramJar;

    @Option(
        name = "experimental_use_rtxt_from_merged_resources",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.CHANGES_INPUTS},
        help = "Use R.txt from the merging action, instead of from the validation action.")
    public boolean useRTxtFromMergedResources;

    @Option(
        name = "legacy_main_dex_list_generator",
        // TODO(b/147692286): Update this default value to R8's GenerateMainDexList binary after
        // migrating usage.
        defaultValue = "null",
        converter = LabelConverter.class,
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "Specifies a binary to use to generate the list of classes that must be in the main"
                + " dex when compiling legacy multidex.")
    public Label legacyMainDexListGenerator;

    @Option(
        name = "experimental_disable_instrumentation_manifest_merge",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
        help =
            "Disables manifest merging when an android_binary has instruments set (i.e. is used "
                + "for instrumentation testing).")
    public boolean disableInstrumentationManifestMerging;

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
      host.dexoptsSupportedInDexSharder = dexoptsSupportedInDexSharder;
      host.useWorkersWithDexbuilder = useWorkersWithDexbuilder;
      host.manifestMerger = manifestMerger;
      host.manifestMergerOrder = manifestMergerOrder;
      host.allowAndroidLibraryDepsWithoutSrcs = allowAndroidLibraryDepsWithoutSrcs;
      host.oneVersionEnforcementUseTransitiveJarsForBinaryUnderTest =
          oneVersionEnforcementUseTransitiveJarsForBinaryUnderTest;
      host.persistentBusyboxTools = persistentBusyboxTools;

      host.incompatibleProhibitAapt1 = incompatibleProhibitAapt1;

      // Unless the build was started from an Android device, host means MAIN.
      host.configurationDistinguisher = ConfigurationDistinguisher.MAIN;
      return host;
    }
  }

  /** Configuration loader for the Android fragment. */
  public static class Loader implements ConfigurationFragmentFactory {
    @Override
    public Fragment create(BuildOptions buildOptions) throws InvalidConfigurationException {
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
  private final ConfigurationDistinguisher configurationDistinguisher;
  private final boolean incrementalDexing;
  private final int incrementalDexingShardsAfterProguard;
  private final boolean incrementalDexingUseDexSharder;
  private final boolean incrementalDexingAfterProguardByDefault;
  private final boolean assumeMinSdkVersion;
  private final ImmutableList<String> dexoptsSupportedInIncrementalDexing;
  private final ImmutableList<String> targetDexoptsThatPreventIncrementalDexing;
  private final ImmutableList<String> dexoptsSupportedInDexMerger;
  private final ImmutableList<String> dexoptsSupportedInDexSharder;
  private final boolean useWorkersWithDexbuilder;
  private final boolean desugarJava8;
  private final boolean desugarJava8Libs;
  private final boolean checkDesugarDeps;
  private final boolean useRexToCompressDexFiles;
  private final boolean allowAndroidLibraryDepsWithoutSrcs;
  private final boolean useAndroidResourceShrinking;
  private final boolean useAndroidResourceCycleShrinking;
  private final boolean useAndroidResourcePathShortening;
  private final boolean useAndroidResourceNameObfuscation;
  private final AndroidManifestMerger manifestMerger;
  private final ManifestMergerOrder manifestMergerOrder;
  private final ApkSigningMethod apkSigningMethod;
  private final boolean useSingleJarApkBuilder;
  private final boolean compressJavaResources;
  private final boolean exportsManifestDefault;
  private final boolean useParallelDex2Oat;
  private final boolean breakBuildOnParallelDex2OatFailure;
  private final boolean omitResourcesInfoProviderFromAndroidBinary;
  private final boolean fixedResourceNeverlinking;
  private final boolean checkForMigrationTag;
  private final boolean oneVersionEnforcementUseTransitiveJarsForBinaryUnderTest;
  private final boolean dataBindingV2;
  private final boolean dataBindingUpdatedArgs;
  private final boolean persistentBusyboxTools;
  private final boolean filterRJarsFromAndroidTest;
  private final boolean removeRClassesFromInstrumentationTestJar;
  private final boolean alwaysFilterDuplicateClassesFromAndroidTest;
  private final boolean filterLibraryJarWithProgramJar;
  private final boolean useRTxtFromMergedResources;
  private final Label legacyMainDexListGenerator;
  private final boolean disableInstrumentationManifestMerging;
  private final boolean incompatibleUseToolchainResolution;

  private AndroidConfiguration(Options options) throws InvalidConfigurationException {
    this.sdk = options.sdk;
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
    this.dexoptsSupportedInDexSharder = ImmutableList.copyOf(options.dexoptsSupportedInDexSharder);
    this.useWorkersWithDexbuilder = options.useWorkersWithDexbuilder;
    this.desugarJava8 = options.desugarJava8;
    this.desugarJava8Libs = options.desugarJava8Libs;
    this.checkDesugarDeps = options.checkDesugarDeps;
    this.allowAndroidLibraryDepsWithoutSrcs = options.allowAndroidLibraryDepsWithoutSrcs;
    this.useAndroidResourceShrinking =
        options.useAndroidResourceShrinking || options.useExperimentalAndroidResourceShrinking;
    this.useAndroidResourceCycleShrinking = options.useAndroidResourceCycleShrinking;
    this.useAndroidResourcePathShortening = options.useAndroidResourcePathShortening;
    this.useAndroidResourceNameObfuscation = options.useAndroidResourceNameObfuscation;
    this.manifestMerger = options.manifestMerger;
    this.manifestMergerOrder = options.manifestMergerOrder;
    this.apkSigningMethod = options.apkSigningMethod;
    this.useSingleJarApkBuilder = options.useSingleJarApkBuilder;
    this.useRexToCompressDexFiles = options.useRexToCompressDexFiles;
    this.compressJavaResources = options.compressJavaResources;
    this.exportsManifestDefault = options.exportsManifestDefault;
    this.useParallelDex2Oat = options.useParallelDex2Oat;
    this.breakBuildOnParallelDex2OatFailure = options.breakBuildOnParallelDex2OatFailure;
    this.omitResourcesInfoProviderFromAndroidBinary =
        options.omitResourcesInfoProviderFromAndroidBinary;
    this.fixedResourceNeverlinking = options.fixedResourceNeverlinking;
    // use --incompatible_disable_native_android_rules, and also the old flag for backwards
    // compatibility
    this.checkForMigrationTag = options.checkForMigrationTag || options.disableNativeAndroidRules;
    this.oneVersionEnforcementUseTransitiveJarsForBinaryUnderTest =
        options.oneVersionEnforcementUseTransitiveJarsForBinaryUnderTest;
    this.dataBindingV2 = options.dataBindingV2;
    this.dataBindingUpdatedArgs = options.dataBindingUpdatedArgs;
    this.persistentBusyboxTools = options.persistentBusyboxTools;
    this.filterRJarsFromAndroidTest = options.filterRJarsFromAndroidTest;
    this.removeRClassesFromInstrumentationTestJar =
        options.removeRClassesFromInstrumentationTestJar;
    this.alwaysFilterDuplicateClassesFromAndroidTest =
        options.alwaysFilterDuplicateClassesFromAndroidTest;
    this.filterLibraryJarWithProgramJar = options.filterLibraryJarWithProgramJar;
    this.useRTxtFromMergedResources = options.useRTxtFromMergedResources;
    this.legacyMainDexListGenerator = options.legacyMainDexListGenerator;
    this.disableInstrumentationManifestMerging = options.disableInstrumentationManifestMerging;
    this.incompatibleUseToolchainResolution = options.incompatibleUseToolchainResolution;

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

  @Override
  public String getCpu() {
    return cpu;
  }

  @StarlarkConfigurationField(
      name = "android_sdk_label",
      doc = "Returns the target denoted by the value of the --android_sdk flag",
      defaultLabel = AndroidRuleClasses.DEFAULT_SDK,
      defaultInToolRepository = true)
  public Label getSdk() {
    return sdk;
  }

  /** Returns whether to use incremental dexing. */
  @Override
  public boolean useIncrementalDexing() {
    return incrementalDexing;
  }

  /** Returns whether to process proguarded Android binaries with incremental dexing tools. */
  @Override
  public int incrementalDexingShardsAfterProguard() {
    return incrementalDexingShardsAfterProguard;
  }

  /** Whether to use a separate tool to shard classes before merging them into final dex files. */
  @Override
  public boolean incrementalDexingUseDexSharder() {
    return incrementalDexingUseDexSharder;
  }

  /** Whether to use incremental dexing to build proguarded binaries by default. */
  @Override
  public boolean incrementalDexingAfterProguardByDefault() {
    return incrementalDexingAfterProguardByDefault;
  }

  /**
   * Returns true if an -assumevalues should be generated for Proguard based on the minSdkVersion of
   * the merged AndroidManifest.
   */
  @Override
  public boolean assumeMinSdkVersion() {
    return assumeMinSdkVersion;
  }

  /** dx flags supported in incremental dexing actions. */
  @Override
  public ImmutableList<String> getDexoptsSupportedInIncrementalDexing() {
    return dexoptsSupportedInIncrementalDexing;
  }

  /** dx flags supported in dexmerger actions. */
  @Override
  public ImmutableList<String> getDexoptsSupportedInDexMerger() {
    return dexoptsSupportedInDexMerger;
  }

  /** dx flags supported in dexmerger actions. */
  public ImmutableList<String> getDexoptsSupportedInDexSharder() {
    return dexoptsSupportedInDexSharder;
  }

  /**
   * Incremental dexing must not be used for binaries that list any of these flags in their {@code
   * dexopts} attribute.
   */
  @Override
  public ImmutableList<String> getTargetDexoptsThatPreventIncrementalDexing() {
    return targetDexoptsThatPreventIncrementalDexing;
  }

  /** Whether to assume the dexbuilder tool supports local worker mode. */
  @Override
  public boolean useWorkersWithDexbuilder() {
    return useWorkersWithDexbuilder;
  }

  @Override
  public boolean desugarJava8() {
    return desugarJava8;
  }

  @Override
  public boolean desugarJava8Libs() {
    return desugarJava8Libs;
  }

  @Override
  public boolean checkDesugarDeps() {
    return checkDesugarDeps;
  }

  @Override
  public boolean useRexToCompressDexFiles() {
    return useRexToCompressDexFiles;
  }

  public boolean allowSrcsLessAndroidLibraryDeps(RuleContext ruleContext) {
    return allowAndroidLibraryDepsWithoutSrcs
        && Allowlist.isAvailable(ruleContext, "allow_deps_without_srcs");
  }

  @Override
  public boolean useAndroidResourceShrinking() {
    return useAndroidResourceShrinking;
  }

  @Override
  public boolean useAndroidResourceCycleShrinking() {
    return useAndroidResourceCycleShrinking;
  }

  @Override
  public boolean useAndroidResourcePathShortening() {
    return useAndroidResourcePathShortening;
  }

  @Override
  public boolean useAndroidResourceNameObfuscation() {
    return useAndroidResourceNameObfuscation;
  }

  public AndroidManifestMerger getManifestMerger() {
    return manifestMerger;
  }

  public ManifestMergerOrder getManifestMergerOrder() {
    return manifestMergerOrder;
  }

  public ApkSigningMethod getApkSigningMethod() {
    return apkSigningMethod;
  }

  @Override
  public boolean apkSigningMethodV1() {
    return apkSigningMethod.signV1();
  }

  @Override
  public boolean apkSigningMethodV2() {
    return apkSigningMethod.signV2();
  }

  @Override
  public boolean useSingleJarApkBuilder() {
    return useSingleJarApkBuilder;
  }

  @Override
  public boolean useParallelDex2Oat() {
    return useParallelDex2Oat;
  }

  @Override
  public boolean breakBuildOnParallelDex2OatFailure() {
    return breakBuildOnParallelDex2OatFailure;
  }

  @Override
  public boolean compressJavaResources() {
    return compressJavaResources;
  }

  @Override
  public boolean getExportsManifestDefault() {
    return exportsManifestDefault;
  }

  @Override
  public boolean omitResourcesInfoProviderFromAndroidBinary() {
    return this.omitResourcesInfoProviderFromAndroidBinary;
  }

  @Override
  public boolean fixedResourceNeverlinking() {
    return this.fixedResourceNeverlinking;
  }

  @Override
  public boolean checkForMigrationTag() {
    return checkForMigrationTag;
  }

  @Override
  public boolean getOneVersionEnforcementUseTransitiveJarsForBinaryUnderTest() {
    return oneVersionEnforcementUseTransitiveJarsForBinaryUnderTest;
  }

  @Override
  public boolean useDataBindingV2() {
    return dataBindingV2;
  }

  @Override
  public boolean useDataBindingUpdatedArgs() {
    return dataBindingUpdatedArgs;
  }

  @Override
  public boolean persistentBusyboxTools() {
    return persistentBusyboxTools;
  }

  @Override
  public boolean incompatibleUseToolchainResolution() {
    return incompatibleUseToolchainResolution;
  }

  @Override
  public String getOutputDirectoryName() {
    return configurationDistinguisher.suffix;
  }

  public boolean filterRJarsFromAndroidTest() {
    return filterRJarsFromAndroidTest;
  }

  public boolean removeRClassesFromInstrumentationTestJar() {
    return removeRClassesFromInstrumentationTestJar;
  }

  public boolean alwaysFilterDuplicateClassesFromAndroidTest() {
    return alwaysFilterDuplicateClassesFromAndroidTest;
  }

  public boolean filterLibraryJarWithProgramJar() {
    return filterLibraryJarWithProgramJar;
  }

  boolean useRTxtFromMergedResources() {
    return useRTxtFromMergedResources;
  }

  public boolean disableInstrumentationManifestMerging() {
    return disableInstrumentationManifestMerging;
  }

  /** Returns the label provided with --legacy_main_dex_list_generator, if any. */
  // TODO(b/147692286): Move R8's main dex list tool into tool repository.
  @StarlarkConfigurationField(
      name = "legacy_main_dex_list_generator",
      doc = "Returns the label provided with --legacy_main_dex_list_generator, if any.")
  @Nullable
  public Label getLegacyMainDexListGenerator() {
    return legacyMainDexListGenerator;
  }
}
