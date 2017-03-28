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

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.EmptyToNullLabelConverter;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.LabelConverter;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.DynamicMode;
import com.google.devtools.build.lib.rules.cpp.CppOptions.DynamicModeConverter;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.List;
import java.util.Set;

/**
 * Configuration fragment for Android rules.
 */
@Immutable
public class AndroidConfiguration extends BuildConfiguration.Fragment {

  /**
   * Converter for {@link com.google.devtools.build.lib.rules.android.AndroidConfiguration.ConfigurationDistinguisher}
   */
  public static final class ConfigurationDistinguisherConverter
      extends EnumConverter<ConfigurationDistinguisher> {
    public ConfigurationDistinguisherConverter() {
      super(ConfigurationDistinguisher.class, "Android configuration distinguisher");
    }
  }

  /**
   * Converter for {@link ApkSigningMethod}.
   */
  public static final class ApkSigningMethodConverter extends EnumConverter<ApkSigningMethod> {
    public ApkSigningMethodConverter() {
      super(ApkSigningMethod.class, "apk signing method");
    }
  }

  /**
   * Converter for a set of {@link AndroidBinaryType}s.
   */
  public static final class AndroidBinaryTypesConverter
      implements Converter<Set<AndroidBinaryType>> {

    private final EnumConverter<AndroidBinaryType> elementConverter =
        new EnumConverter<AndroidBinaryType>(AndroidBinaryType.class, "Android binary type") {};
    private final Splitter splitter = Splitter.on(',').omitEmptyStrings().trimResults();

    public AndroidBinaryTypesConverter() {}

    @Override
    public ImmutableSet<AndroidBinaryType> convert(String input) throws OptionsParsingException {
      if ("all".equals(input)) {
        return ImmutableSet.copyOf(AndroidBinaryType.values());
      }
      ImmutableSet.Builder<AndroidBinaryType> result = ImmutableSet.builder();
      for (String opt : splitter.split(input)) {
        result.add(elementConverter.convert(opt));
      }
      return result.build();
    }

    @Override
    public String getTypeDescription() {
      return "comma-separated list of: " + elementConverter.getTypeDescription();
    }
  }

  /**
   * Converter for {@link AndroidManifestMerger}
   */
  public static final class AndroidManifestMergerConverter
      extends EnumConverter<AndroidManifestMerger> {
    public AndroidManifestMergerConverter() {
      super(AndroidManifestMerger.class, "android manifest merger");
    }
  }

  /**
   * Value used to avoid multiple configurations from conflicting.
   *
   * <p>This is set to {@code ANDROID} in Android configurations and to {@code MAIN} otherwise. This
   * influences the output directory name: if it didn't, an Android and a non-Android configuration
   * would conflict if they had the same toolchain identifier.
   *
   * <p>Note that this is not just a theoretical concern: even if {@code --crosstool_top} and
   * {@code --android_crosstool_top} point to different labels, they may end up being redirected to
   * the same thing, and this is exactly what happens on OSX X.
   */
  public enum ConfigurationDistinguisher {
    MAIN(null),
    ANDROID("android");

    private final String suffix;

    private ConfigurationDistinguisher(String suffix) {
      this.suffix = suffix;
    }
  }

  /** Types of android binaries as {@link AndroidBinary#dex} distinguishes them. */
  public enum AndroidBinaryType {
    MONODEX, MULTIDEX_UNSHARDED, MULTIDEX_SHARDED
  }

  /**
   * Which APK signing method to use with the debug key for rules that build APKs.
   *
   * <ul>
   * <li>V1 uses the apksigner attribute from the android_sdk and signs the APK as a JAR.
   * <li>V2 uses the apksigner attribute from the android_sdk and signs the APK according to the APK
   * Signing Schema V2 that is only supported on Android N and later.
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

    /** Wheter to sign the APK with the apksigner tool with APK Signature Schema V2. */
    public boolean signV2() {
      return signV2;
    }
  }

  /** Types of android manifest mergers. */
  public enum AndroidManifestMerger {
    LEGACY,
    ANDROID;

    public static List<String> getAttributeValues() {
      return ImmutableList.of(LEGACY.name().toLowerCase(), ANDROID.name().toLowerCase(),
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

  /**
   * Android configuration options.
   */
  public static class Options extends FragmentOptions {
    @Option(name = "Android configuration distinguisher",
        defaultValue = "MAIN",
        converter = ConfigurationDistinguisherConverter.class,
        category = "internal")
    public ConfigurationDistinguisher configurationDistinguisher;

    // For deploying incremental installation of native libraries. Do not use on the command line.
    // The idea is that once this option works, we'll flip the default value in a config file, then
    // once it is proven that it works, remove it from Bazel and said config file.
    @Option(name = "android_incremental_native_libs",
        defaultValue = "false",
        category = "undocumented")
    public boolean incrementalNativeLibs;

    @Option(name = "android_crosstool_top",
        defaultValue = "//external:android/crosstool",
        category = "semantics",
        converter = EmptyToNullLabelConverter.class,
        help = "The location of the C++ compiler used for Android builds.")
    public Label androidCrosstoolTop;

    @Option(name = "android_cpu",
        defaultValue = "armeabi",
        category = "semantics",
        help = "The Android target CPU.")
    public String cpu;

    @Option(
      name = "android_compiler",
      defaultValue = "null",
      category = "semantics",
      help = "The Android target compiler."
    )
    public String cppCompiler;

    @Option(
      name = "android_dynamic_mode",
      defaultValue = "off",
      converter = DynamicModeConverter.class,
      category = "undocumented",
      help =
        "Determines whether C++ deps of Android rules will be linked dynamically when a cc_binary "
            + "does not explicitly create a shared library. 'default' means blaze will choose "
            + "whether to link dynamically.  'fully' means all libraries will be linked "
            + "dynamically. 'off' means that all libraries will be linked in mostly static mode."
    )
    public DynamicMode dynamicMode;

    // Label of filegroup combining all Android tools used as implicit dependencies of
    // android_* rules
    @Option(name = "android_sdk",
            defaultValue = "@bazel_tools//tools/android:sdk",
            category = "version",
            converter = LabelConverter.class,
            help = "Specifies Android SDK/platform that is used to build Android applications.")
    public Label sdk;

    // TODO(bazel-team): Maybe merge this with --android_cpu above.
    @Option(name = "fat_apk_cpu",
            converter = Converters.CommaSeparatedOptionListConverter.class,
            defaultValue = "armeabi-v7a",
            category = "semantics",
            help = "Setting this option enables fat APKs, which contain native binaries for all "
                + "specified target architectures, e.g., --fat_apk_cpu=x86,armeabi-v7a. If this "
                + "flag is specified, then --android_cpu is ignored for dependencies of "
                + "android_binary rules.")
    public List<String> fatApkCpus;

    @Option(name = "experimental_android_use_jack_for_dexing",
        defaultValue = "false",
        category = "semantics",
        help = "Switches to the Jack and Jill toolchain for dexing instead of javac and dx.")
    public boolean useJackForDexing;

    @Option(name = "experimental_android_jack_sanity_checks",
        defaultValue = "false",
        category = "semantics",
        help = "Enables sanity checks for Jack and Jill compilation.")
    public boolean jackSanityChecks;

    // For desugaring lambdas when compiling Java 8 sources. Do not use on the command line.
    // The idea is that once this option works, we'll flip the default value in a config file, then
    // once it is proven that it works, remove it from Bazel and said config file.
    @Option(name = "experimental_desugar_for_android",
        defaultValue = "false",
        category = "undocumented",
        help = "Whether to desugar Java 8 bytecode before dexing.")
    public boolean desugarJava8;

    @Option(name = "incremental_dexing",
        defaultValue = "false",
        category = "semantics",
        implicitRequirements = "--noexperimental_android_use_jack_for_dexing",
        help = "Does most of the work for dexing separately for each Jar file.  Incompatible with "
            + "Jack and Jill.")
    public boolean incrementalDexing;

    @Option(name = "host_incremental_dexing",
        defaultValue = "false",
        category = "hidden",
        help = "This flag is deprecated in favor of applying --incremental_dexing to both host "
            + "and target configuration.  This flag will be removed in a future release.")
    public boolean hostIncrementalDexing;

    // Do not use on the command line.
    // The idea is that this option lets us gradually turn on incremental dexing for different
    // binaries.  Users should rely on --noincremental_dexing to turn it off.
    @Option(name = "incremental_dexing_binary_types",
        defaultValue = "multidex_sharded",
        category = "undocumented",
        converter = AndroidBinaryTypesConverter.class,
        implicitRequirements = "--incremental_dexing",
        help = "Kinds of binaries to incrementally dex if --incremental_dexing is true.")
    public Set<AndroidBinaryType> incrementalDexingBinaries;

    /**
     * Whether to look for incrementally dex protos built with java_lite_proto_library. Once this
     * option works, we'll flip the default value in a config file, then once it is proven that it
     * works, remove it from Bazel and said config file.
     */
    @Option(
      name = "experimental_incremental_dexing_for_lite_protos",
      defaultValue = "false",
      category = "experimental",
      help = "Do not use.")
    public boolean incrementalDexingForLiteProtos;

    /**
     * Whether to error out when we find Jar files when building binaries that weren't converted to
     * a dex archive. This option will soon be removed from Bazel.
     */
    @Option(
      name = "experimental_incremental_dexing_error_on_missed_jars",
      defaultValue = "true",
      category = "experimental",
      help = "Do not use.")
    public boolean incrementalDexingErrorOnMissedJars;

    @Option(name = "non_incremental_per_target_dexopts",
        converter = Converters.CommaSeparatedOptionListConverter.class,
        defaultValue = "--no-locals",
        category = "semantics",
        help = "dx flags that that prevent incremental dexing for binary targets that list any of "
            + "the flags listed here in their 'dexopts' attribute, which are ignored with "
            + "incremental dexing (superseding --dexopts_supported_in_incremental_dexing).  "
            + "Defaults to --no-locals for safety but can in general be used "
            + "to make sure the listed dx flags are honored, with additional build latency.  "
            + "Please notify us if you find yourself needing this flag.")
    public List<String> nonIncrementalPerTargetDexopts;

    // Do not use on the command line.
    // This flag is intended to be updated as we add supported flags to the incremental dexing tools
    @Option(name = "dexopts_supported_in_incremental_dexing",
        converter = Converters.CommaSeparatedOptionListConverter.class,
        defaultValue = "",
        category = "hidden",
        help = "dx flags supported in incremental dexing.")
    public List<String> dexoptsSupportedInIncrementalDexing;

    @Option(
      name = "experimental_android_rewrite_dexes_with_rex",
      defaultValue = "false",
      category = "undocumented",
      help = "use rex tool to rewrite dex files"
    )
    public boolean useRexToCompressDexFiles;

    @Option(name = "experimental_allow_android_library_deps_without_srcs",
        defaultValue = "true",
        category = "undocumented",
        help = "Flag to help transition from allowing to disallowing srcs-less android_library"
            + " rules with deps. The depot needs to be cleaned up to roll this out by default.")
    public boolean allowAndroidLibraryDepsWithoutSrcs;

    @Option(name = "experimental_android_resource_shrinking",
        defaultValue = "false",
        category = "undocumented",
        help = "Enables resource shrinking for android_binary APKs that use ProGuard.")
    public boolean useExperimentalAndroidResourceShrinking;

    @Option(name = "android_resource_shrinking",
        defaultValue = "false",
        category = "semantics",
        help = "Enables resource shrinking for android_binary APKs that use ProGuard.")
    public boolean useAndroidResourceShrinking;

    @Option(name = "android_manifest_merger",
        defaultValue = "android",
        category = "semantics",
        converter = AndroidManifestMergerConverter.class,
        help = "Selects the manifest merger to use for android_binary rules. Flag to help the"
            + "transition to the Android manifest merger from the legacy merger.")
    public AndroidManifestMerger manifestMerger;

    // Do not use on the command line.
    @Option(name = "experimental_use_parallel_android_resource_processing",
      defaultValue = "true",
      category = "undocumented",
      help = "Process android_library resources with higher parallelism. Generates library "
              + "R classes from a merge action, separately from aapt.")
    public boolean useParallelResourceProcessing;

    @Option(name = "apk_signing_method",
        converter = ApkSigningMethodConverter.class,
        defaultValue = "v1_v2",
        category = "undocumented",
        help = "Implementation to use to sign APKs")
    public ApkSigningMethod apkSigningMethod;

    @Option(name = "use_singlejar_apkbuilder",
        defaultValue = "false",
        category = "undocumented",
        help = "Build Android APKs with SingleJar.")
    public boolean useSingleJarApkBuilder;

    @Option(name = "experimental_android_use_singlejar_for_multidex",
        defaultValue = "false",
        category = "undocumented",
        help = "Use SingleJar for multidex dex extraction.")
    public boolean useSingleJarForMultidex;

    @Option(name = "experimental_android_use_resource_prefiltering",
      defaultValue = "false",
      category = "undocumented",
      help = "When building android_binary targets, apply resource_configuration_filters during"
          + " the analysis phase. Normally, this filtering doesn't occur until near the end of "
          + "resource processing in the execution phase. Prefiltering resources improves "
          + "performance for binaries with resource_configuration_filters and lots of resources."
    )
    public boolean useResourcePrefiltering;

    @Option(name = "use_singlejar_for_proguard_libraryjars",
        defaultValue = "false",
        category = "undocumented",
        help = "Use SingleJar to combine all ProGuard library jars.")
    public boolean useSingleJarForProguardLibraryJars;

    @Override
    public void addAllLabels(Multimap<String, Label> labelMap) {
      if (androidCrosstoolTop != null) {
        labelMap.put("android_crosstool_top", androidCrosstoolTop);
      }

      labelMap.put("android_sdk", sdk);
    }

    @Override
    public FragmentOptions getHost(boolean fallback) {
      Options host = (Options) super.getHost(fallback);
      host.androidCrosstoolTop = androidCrosstoolTop;
      host.sdk = sdk;
      host.fatApkCpus = ImmutableList.<String>of(); // Fat APK archs don't apply to the host.

      host.desugarJava8 = desugarJava8;
      host.incrementalDexing = incrementalDexing;
      host.incrementalDexingBinaries = incrementalDexingBinaries;
      host.incrementalDexingForLiteProtos = incrementalDexingForLiteProtos;
      host.incrementalDexingErrorOnMissedJars = incrementalDexingErrorOnMissedJars;
      host.nonIncrementalPerTargetDexopts = nonIncrementalPerTargetDexopts;
      host.dexoptsSupportedInIncrementalDexing = dexoptsSupportedInIncrementalDexing;
      host.manifestMerger = manifestMerger;
      return host;
    }

    @Override
    public ImmutableList<String> getDefaultsRules() {
      return ImmutableList.of("android_tools_defaults_jar(name = 'android_jar')");
    }

    @Override
    public List<SplitTransition<BuildOptions>> getPotentialSplitTransitions() {
      return ImmutableList.of(AndroidRuleClasses.ANDROID_SPLIT_TRANSITION);
    }
  }

  /**
   * Configuration loader for the Android fragment.
   */
  public static class Loader implements ConfigurationFragmentFactory {
    @Override
    public Fragment create(ConfigurationEnvironment env, BuildOptions buildOptions)
        throws InvalidConfigurationException, InterruptedException {
      AndroidConfiguration.Options androidOptions =
          buildOptions.get(AndroidConfiguration.Options.class);
      return new AndroidConfiguration(buildOptions.get(Options.class), androidOptions.sdk);
    }

    @Override
    public Class<? extends Fragment> creates() {
      return AndroidConfiguration.class;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.<Class<? extends FragmentOptions>>of(Options.class);
    }
  }

  private final Label sdk;
  private final String cpu;
  private final boolean incrementalNativeLibs;
  private final ConfigurationDistinguisher configurationDistinguisher;
  private final boolean useJackForDexing;
  private final boolean jackSanityChecks;
  private final ImmutableSet<AndroidBinaryType> incrementalDexingBinaries;
  private final boolean incrementalDexingForLiteProtos;
  private final boolean incrementalDexingErrorOnMissedJars;
  private final ImmutableList<String> dexoptsSupportedInIncrementalDexing;
  private final ImmutableList<String> targetDexoptsThatPreventIncrementalDexing;
  private final boolean desugarJava8;
  private final boolean useRexToCompressDexFiles;
  private final boolean allowAndroidLibraryDepsWithoutSrcs;
  private final boolean useAndroidResourceShrinking;
  private final boolean useParallelResourceProcessing;
  private final AndroidManifestMerger manifestMerger;
  private final ApkSigningMethod apkSigningMethod;
  private final boolean useSingleJarApkBuilder;
  private final boolean useSingleJarForMultidex;
  private final boolean useResourcePrefiltering;
  private final boolean useSingleJarForProguardLibraryJars;

  AndroidConfiguration(Options options, Label androidSdk) {
    this.sdk = androidSdk;
    this.incrementalNativeLibs = options.incrementalNativeLibs;
    this.cpu = options.cpu;
    this.configurationDistinguisher = options.configurationDistinguisher;
    this.useJackForDexing = options.useJackForDexing;
    this.jackSanityChecks = options.jackSanityChecks;
    if (options.incrementalDexing) {
      this.incrementalDexingBinaries = ImmutableSet.copyOf(options.incrementalDexingBinaries);
    } else {
      this.incrementalDexingBinaries = ImmutableSet.of();
    }
    this.incrementalDexingForLiteProtos = options.incrementalDexingForLiteProtos;
    this.incrementalDexingErrorOnMissedJars = options.incrementalDexingErrorOnMissedJars;
    this.dexoptsSupportedInIncrementalDexing =
        ImmutableList.copyOf(options.dexoptsSupportedInIncrementalDexing);
    this.targetDexoptsThatPreventIncrementalDexing =
        ImmutableList.copyOf(options.nonIncrementalPerTargetDexopts);
    this.desugarJava8 = options.desugarJava8;
    this.allowAndroidLibraryDepsWithoutSrcs = options.allowAndroidLibraryDepsWithoutSrcs;
    this.useAndroidResourceShrinking = options.useAndroidResourceShrinking
        || options.useExperimentalAndroidResourceShrinking;
    this.useParallelResourceProcessing = options.useParallelResourceProcessing;
    this.manifestMerger = options.manifestMerger;
    this.apkSigningMethod = options.apkSigningMethod;
    this.useSingleJarApkBuilder = options.useSingleJarApkBuilder;
    this.useSingleJarForMultidex = options.useSingleJarForMultidex;
    this.useSingleJarForProguardLibraryJars = options.useSingleJarForProguardLibraryJars;
    this.useRexToCompressDexFiles = options.useRexToCompressDexFiles;
    this.useResourcePrefiltering = options.useResourcePrefiltering;
  }

  public String getCpu() {
    return cpu;
  }

  public Label getSdk() {
    return sdk;
  }

  /**
   * Returns true if Jack should be used in place of javac/dx for Android compilation.
   */
  public boolean isJackUsedForDexing() {
    return useJackForDexing;
  }

  /**
   * Returns true if Jack sanity checks should be enabled. Only relevant if isJackUsedForDexing()
   * also returns true.
   */
  public boolean isJackSanityChecked() {
    return jackSanityChecks;
  }

  public boolean useIncrementalNativeLibs() {
    return incrementalNativeLibs;
  }

  /**
   * Returns when to use incremental dexing using {@link DexArchiveProvider}.  Note this is disabled
   * if {@link #isJackUsedForDexing()}.
   */
  public ImmutableSet<AndroidBinaryType> getIncrementalDexingBinaries() {
    return isJackUsedForDexing() ? ImmutableSet.<AndroidBinaryType>of() : incrementalDexingBinaries;
  }

  /**
   * Returns whether to look for Jars produced by {@code JavaLiteProtoAspect}.
   */
  public boolean incrementalDexingForLiteProtos() {
    return incrementalDexingForLiteProtos;
  }

  /**
   * Returns whether to report an error when Jars that weren't converted to dex archives are part
   * of an android binary.
   */
  public boolean incrementalDexingErrorOnMissedJars() {
    return incrementalDexingErrorOnMissedJars;
  }

  /**
   * dx flags supported in incremental dexing actions.
   */
  public ImmutableList<String> getDexoptsSupportedInIncrementalDexing() {
    return dexoptsSupportedInIncrementalDexing;
  }

  /**
   * Regardless of {@link #getIncrementalDexingBinaries}, incremental dexing must not be used for
   * binaries that list any of these flags in their {@code dexopts} attribute.
   */
  public ImmutableList<String> getTargetDexoptsThatPreventIncrementalDexing() {
    return targetDexoptsThatPreventIncrementalDexing;
  }

  public boolean desugarJava8() {
    return desugarJava8;
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

  public boolean useParallelResourceProcessing() {
    return useParallelResourceProcessing;
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

  public boolean useSingleJarForMultidex() {
    return useSingleJarForMultidex;
  }

  public boolean useResourcePrefiltering() {
    return useResourcePrefiltering;
  }

  public boolean useSingleJarForProguardLibraryJars() {
    return useSingleJarForProguardLibraryJars;
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
