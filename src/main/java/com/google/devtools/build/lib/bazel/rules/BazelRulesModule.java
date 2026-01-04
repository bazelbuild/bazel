// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.LabelConverter;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.exec.ModuleActionContextRegistry;
import com.google.devtools.build.lib.rules.java.JavaCompileActionContext;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.TriState;
import java.util.List;

/** Module implementing the rule set of Bazel. */
public final class BazelRulesModule extends BlazeModule {

  /**
   * This is where deprecated options used by both Bazel and Blaze but only needed for the build
   * command go to die.
   *
   * <p>To deprecate Bazel-only build command options, use {@link BazelBuildGraveyardOptions}.
   *
   * <p>To deprecate Bazel+Blaze options common to all commands, use {@link
   * AllCommandGraveyardOptions}.
   */
  public static class BuildGraveyardOptions extends OptionsBase {

    @Deprecated
    @Option(
        name = "build_python_zip",
        defaultValue = "auto",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public TriState buildPythonZip;

    @Deprecated
    @Option(
        name = "incompatible_default_to_explicit_init_py",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public boolean incompatibleDefaultToExplicitInitPy;

    @Deprecated
    @Option(
        name = "python_native_rules_allowlist",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        converter = LabelConverter.class,
        help = "Deprecated. No-op.")
    public Label nativeRulesAllowlist;

    @Deprecated
    @Option(
        name = "incompatible_python_disallow_native_rules",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public boolean disallowNativeRules;

    @Deprecated
    @Option(
        name = "incompatible_remove_ctx_py_fragment",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public boolean disablePyFragment;

    @Deprecated
    @Option(
        name = "incompatible_use_python_toolchains",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public boolean incompatibleUsePythonToolchains;

    @Deprecated
    @Option(
        name = "experimental_starlark_cc_import",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public boolean experimentalStarlarkCcImport;

    @Deprecated
    @Option(
        name = "experimental_platform_cc_test",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public boolean experimentalPlatformCcTest;

    @Deprecated
    @Option(
        name = "j2objc_dead_code_removal",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public boolean removeDeadCode;

    @Deprecated
    @Option(
        name = "experimental_proto_extra_actions",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.EXPERIMENTAL, OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public boolean experimentalProtoExtraActions;

    @Deprecated
    @Option(
        name = "enable_fdo_profile_absolute_path",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public boolean enableFdoProfileAbsolutePath;

    @Deprecated
    @Option(
        name = "experimental_genquery_use_graphless_query",
        defaultValue = "auto",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public TriState useGraphlessQuery;

    @Deprecated
    @Option(
        name = "use_top_level_targets_for_symlinks",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public boolean useTopLevelTargetsForSymlinks;

    @Deprecated
    @Option(
        name = "experimental_skyframe_prepare_analysis",
        deprecationWarning = "This flag is a no-op and will be deleted in a future release.",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public boolean skyframePrepareAnalysis;

    @Deprecated
    @Option(
        name = "incompatible_disable_legacy_flags_cc_toolchain_api",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public boolean disableLegacyFlagsCcToolchainApi;

    @Deprecated
    @Option(
        name = "incompatible_enable_profile_by_default",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean enableProfileByDefault;

    @Deprecated
    @Option(
        name = "incompatible_override_toolchain_transition",
        defaultValue = "true",
        deprecationWarning = "This is now always set, please remove this flag.",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = OptionEffectTag.NO_OP,
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated, this is no longer in use and should be removed.")
    public boolean overrideToolchainTransition;

    @Deprecated
    @Option(
        name = "experimental_parse_headers_skipped_if_corresponding_srcs_found",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean parseHeadersSkippedIfCorrespondingSrcsFound;

    @Deprecated
    @Option(
        name = "experimental_worker_as_resource",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op, will be removed soon.")
    public boolean workerAsResource;

    @Deprecated
    @Option(
        name = "high_priority_workers",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op, will be removed soon.",
        allowMultiple = true)
    public List<String> highPriorityWorkers;

    @Deprecated
    @Option(
        name = "target_platform_fallback",
        defaultValue = "",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "This option is deprecated and has no effect.")
    public String targetPlatformFallback;

    @Deprecated
    @Option(
        name = "incompatible_auto_configure_host_platform",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "This option is deprecated and has no effect.")
    public boolean autoConfigureHostPlatform;

    @Deprecated
    @Option(
        name = "experimental_require_availability_info",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        defaultValue = "false",
        help = "Deprecated no-op.")
    public boolean requireAvailabilityInfo;

    @Deprecated
    @Option(
        name = "experimental_collect_local_action_metrics",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated no-op.")
    public boolean collectLocalExecutionStatistics;

    @Deprecated
    @Option(
        name = "experimental_collect_local_sandbox_action_metrics",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated no-op.")
    public boolean collectLocalSandboxExecutionStatistics;

    @Deprecated
    @Option(
        name = "experimental_enable_starlark_doc_extract",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.EXPERIMENTAL, OptionMetadataTag.DEPRECATED},
        help = "Deprecated no-op.")
    public boolean enableBzlDocDump;

    // TODO(b/274595070): Remove this option.
    @Deprecated
    @Option(
        name = "experimental_parallel_aquery_output",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean parallelAqueryOutput;

    @Deprecated
    @Option(
        name = "experimental_show_artifacts",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated no-op.")
    public boolean showArtifacts;

    @Deprecated
    @Option(
        name = "announce",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.",
        deprecationWarning = "This option is now deprecated and is a no-op")
    public boolean announce;

    @Deprecated
    @Option(
        name = "action_cache_store_output_metadata",
        oldName = "experimental_action_cache_store_output_metadata",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "no-op")
    public boolean actionCacheStoreOutputMetadata;

    @Deprecated
    @Option(
        name = "discard_actions_after_execution",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        effectTags = {OptionEffectTag.NO_OP},
        help = "This option is deprecated and has no effect.")
    public boolean discardActionsAfterExecution;

    @Deprecated
    @Option(
        name = "defer_param_files",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "This option is deprecated and has no effect and will be removed in the future.")
    public boolean deferParamFiles;

    @Deprecated
    @Option(
        name = "check_fileset_dependencies_recursively",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        deprecationWarning =
            "This flag is a no-op and fileset dependencies are always checked "
                + "to ensure correctness of builds.",
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED})
    public boolean checkFilesetDependenciesRecursively;

    @Deprecated
    @Option(
        name = "experimental_skyframe_native_filesets",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        deprecationWarning = "This flag is a no-op and skyframe-native-filesets is always true.")
    public boolean skyframeNativeFileset;

    @Deprecated
    @Option(
        name = "collapse_duplicate_defines",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "no-op")
    public boolean collapseDuplicateDefines;

    @Deprecated
    @Option(
        name = "incompatible_require_javaplugininfo_in_javacommon",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public boolean requireJavaPluginInfo;

    @Deprecated
    @Option(
        name = "experimental_build_transitive_python_runfiles",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op",
        oldName = "incompatible_build_transitive_python_runfiles")
    public boolean buildTransitiveRunfilesTrees;

    @Deprecated
    @Option(
        name = "experimental_use_new_worker_pool",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public boolean useNewWorkerPool;

    @Deprecated
    @Option(
        name = "host_crosstool_top",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public String hostCrosstoolTop;

    @Deprecated
    @Option(
        name = "experimental_use_semaphore_for_jobs",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean useSemaphoreForJobs;

    // TODO(b/410585542): Remove this once there are no more internal users trying to set it.
    @Deprecated
    @Option(
        name = "experimental_skip_ttvs_for_genquery",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op. Will be removed soon.")
    public boolean skipTtvs;
  }

  /** This is where deprecated Bazel-specific options only used by the build command go to die. */
  public static final class BazelBuildGraveyardOptions extends BuildGraveyardOptions {
    @Deprecated
    @Option(
        name = "python_path",
        defaultValue = "",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public String pythonPath;

    @Deprecated
    @Option(
        name = "experimental_python_import_all_repositories",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public boolean experimentalPythonImportAllRepositories;

    @Deprecated
    @Option(
        name = "python_top",
        defaultValue = "",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public String pythonTop;

    @Deprecated
    @Option(
        name = "incompatible_load_python_rules_from_bzl",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "Deprecated no-op.")
    public boolean loadPythonRulesFromBzl;

    @Deprecated
    @Option(
        name = "incompatible_load_proto_rules_from_bzl",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "Deprecated no-op.")
    public boolean loadProtoRulesFromBzl;

    @Deprecated
    @Option(
        name = "incompatible_load_java_rules_from_bzl",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "Deprecated no-op.")
    public boolean loadJavaRulesFromBzl;

    @Deprecated
    @Option(
        name = "make_variables_source",
        defaultValue = "configuration",
        metadataTags = {OptionMetadataTag.HIDDEN, OptionMetadataTag.DEPRECATED},
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP})
    public String makeVariableSource;

    @Deprecated
    @Option(
        name = "force_ignore_dash_static",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "noop")
    public boolean forceIgnoreDashStatic;

    @Deprecated
    @Option(
        name = "experimental_profile_action_counts",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean enableActionCountProfile;

    @Deprecated
    @Option(
        name = "incompatible_remove_binary_profile",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean removeBinaryProfile;

    @Deprecated
    @Option(
        name = "experimental_post_profile_started_event",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean postProfileStartedEvent;

    @Deprecated
    @Option(
        name = "incompatible_dont_use_javasourceinfoprovider",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public boolean dontUseJavaSourceInfoProvider;

    private static final String ANDROID_FLAG_DEPRECATION =
        "Legacy Android flags have been deprecated. See"
            + " https://blog.bazel.build/2023/11/15/android-platforms.html for details and"
            + " migration directions";

    @Deprecated
    @Option(
        name = "android_sdk",
        defaultValue = "",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.UNKNOWN},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op",
        deprecationWarning = ANDROID_FLAG_DEPRECATION)
    public String sdk;

    @Deprecated
    @Option(
        name = "android_cpu",
        defaultValue = "",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op",
        deprecationWarning = ANDROID_FLAG_DEPRECATION)
    public String cpu;

    @Deprecated
    @Option(
        name = "android_crosstool_top",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op",
        deprecationWarning = ANDROID_FLAG_DEPRECATION)
    public String androidCrosstoolTop;

    @Deprecated
    @Option(
        name = "android_grte_top",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op",
        deprecationWarning = ANDROID_FLAG_DEPRECATION)
    public String androidLibcTopLabel;

    @Deprecated
    @Option(
        name = "incompatible_enable_android_toolchain_resolution",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op",
        deprecationWarning = ANDROID_FLAG_DEPRECATION)
    public boolean incompatibleUseToolchainResolution;

    @Deprecated
    @Option(
        name = "fat_apk_cpu",
        converter = Converters.CommaSeparatedOptionSetConverter.class,
        defaultValue = "armeabi-v7a",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op",
        deprecationWarning = ANDROID_FLAG_DEPRECATION)
    public List<String> fatApkCpus;

    @Deprecated
    @Option(
        name = "incompatible_enable_cgo_toolchain_resolution",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public boolean incompatibleEnableGoToolchainResolution;
  }

  /**
   * This is where deprecated options which need to be available for all commands go to die. If you
   * want to graveyard an all-command option specific to Blaze or Bazel, create a subclass.
   */
  public static final class AllCommandGraveyardOptions extends OptionsBase {

    @Deprecated
    @Option(
        name = "experimental_py_binaries_include_label",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean includeLabelInPyBinariesLinkstamp;

    @Deprecated
    @Option(
        name = "incompatible_python_disable_py2",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean disablePy2;

    @Deprecated
    @Option(
        name = "force_python",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public String forcePython;

    @Deprecated
    @Option(
        name = "host_force_python",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public String hostForcePython;

    @Deprecated
    @Option(
        name = "incompatible_py3_is_default",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean incompatiblePy3IsDefault;

    @Deprecated
    @Option(
        name = "incompatible_py2_outputs_are_suffixed",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean incompatiblePy2OutputsAreSuffixed;

    @Deprecated
    @Option(
        name = "python_version",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public String pythonVersion;

    @Deprecated
    @Option(
        name = "incompatible_remove_old_python_version_api",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean incompatibleRemoveOldPythonVersionApi;

    @Deprecated
    @Option(
        name = "incompatible_allow_python_version_transitions",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean incompatibleAllowPythonVersionTransitions;

    @Deprecated
    @Option(
        name = "incompatible_disallow_legacy_py_provider",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean incompatibleDisallowLegacyPyProvider;

    @Deprecated
    @Option(
        name = "experimental_disallow_legacy_java_toolchain_flags",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean experimentalDisallowLegacyJavaToolchainFlags;

    @Deprecated
    @Option(
        name = "javabase",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public String javabase;

    @Deprecated
    @Option(
        name = "java_toolchain",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public String javaToolchain;

    @Deprecated
    @Option(
        name = "host_java_toolchain",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public String hostJavaToolchain;

    @Deprecated
    @Option(
        name = "host_javabase",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public String hostJavabase;

    @Deprecated
    @Option(
        name = "apple_crosstool_top",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public String appleCrosstoolTop;

    @Deprecated
    @Option(
        name = "legacy_bazel_java_test",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public boolean legacyBazelJavaTest;

    @Deprecated
    @Option(
        name = "strict_deps_java_protos",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public boolean strictDepsJavaProtos;

    @Deprecated
    @Option(
        name = "disallow_strict_deps_for_jpl",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public boolean isDisallowStrictDepsForJpl;

    @Deprecated
    @Option(
        name = "experimental_import_deps_checking",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public String importDepsCheckingLevel;

    @Deprecated
    @Option(
        name = "experimental_allow_runtime_deps_on_neverlink",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public boolean allowRuntimeDepsOnNeverLink;

    @Deprecated
    @Option(
        name = "experimental_limit_android_lint_to_android_constrained_java",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public boolean limitAndroidLintToAndroidCompatible;

    @Deprecated
    @Option(
        name = "experimental_java_header_input_pruning",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public boolean experimentalJavaHeaderInputPruning;

    @Deprecated
    @Option(
        name = "incompatible_dont_collect_native_libraries_in_data",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean dontCollectDataLibraries;

    @Deprecated
    @Option(
        name = "experimental_java_header_compilation_direct_deps",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean javaHeaderCompilationDirectDeps;

    @Deprecated
    @Option(
        name = "jplPropagateCcLinkParamsStore",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public boolean jplPropagateCcLinkParamsStore;

    @Deprecated
    @Option(
        name = "incompatible_require_linker_input_cc_api",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public boolean incompatibleRequireLinkerInputCcApi;

    @Deprecated
    @Option(
        name = "incompatible_depset_for_libraries_to_link_getter",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public boolean incompatibleDepsetForLibrariesToLinkGetter;

    @Deprecated
    @Option(
        name = "legacy_external_runfiles",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public boolean legacyExternalRunfiles;

    @Deprecated
    @Option(
        name = "incompatible_disable_target_provider_fields",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public boolean incompatibleDisableTargetProviderFields;

    @Deprecated
    @Option(
        name = "incompatible_disallow_struct_provider_syntax",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public boolean incompatibleDisallowStructProviderSyntax;

    @Deprecated
    @Option(
        name = "auto_cpu_environment_group",
        defaultValue = "",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.EXPERIMENTAL, OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public String autoCpuEnvironmentGroup;

    @Deprecated
    @Option(
        name = "incompatible_top_level_aspects_require_providers",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        effectTags = {OptionEffectTag.NO_OP},
        help = "No-op")
    public boolean incompatibleTopLevelAspectsRequireProviders;

    @Deprecated
    @Option(
        name = "separate_aspect_deps",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public boolean separateAspectDeps;

    @Deprecated
    @Option(
        name = "incompatible_visibility_private_attributes_at_definition",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public boolean incompatibleVisibilityPrivateAttributesAtDefinition;

    @Deprecated
    @Option(
        name = "incompatible_new_actions_api",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        effectTags = {OptionEffectTag.NO_OP},
        help = "No-op")
    public boolean incompatibleNewActionsApi;

    @Deprecated
    @Option(
        name = "experimental_enable_aspect_hints",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
        metadataTags = {OptionMetadataTag.EXPERIMENTAL, OptionMetadataTag.DEPRECATED})
    public boolean enableAspectHints;

    @Deprecated
    @Option(
        name = "bes_best_effort",
        defaultValue = "false",
        deprecationWarning =
            "BES best effort upload has been removed. The flag has no more "
                + "functionality attached to it and will be removed in a future release.",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public boolean besBestEffort;

    @Deprecated
    @Option(
        name = "experimental_use_event_based_build_completion_status",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        metadataTags = {OptionMetadataTag.EXPERIMENTAL, OptionMetadataTag.DEPRECATED},
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.EXECUTION},
        help = "No-op")
    public boolean useEventBasedBuildCompletionStatus;

    @Deprecated
    @Option(
        name = "experimental_java_proto_add_allowed_public_imports",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.EXPERIMENTAL, OptionMetadataTag.DEPRECATED},
        help = "This flag is a noop and scheduled for removal.")
    public boolean experimentalJavaProtoAddAllowedPublicImports;

    @Deprecated
    @Option(
        name = "java_optimization_mode",
        defaultValue = "",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Do not use.")
    public String javaOptimizationMode;

    @Deprecated
    @Option(
        name = "incompatible_depset_for_java_output_source_jars",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean incompatibleDepsetForJavaOutputSourceJars;

    @Deprecated
    @Option(
        name = "incompatible_use_plus_in_repo_names",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = OptionEffectTag.LOADING_AND_ANALYSIS,
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean incompatibleUsePlusInRepoNames;

    @Deprecated
    @Option(
        name = "enable_bzlmod",
        oldName = "experimental_enable_bzlmod",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = OptionEffectTag.LOADING_AND_ANALYSIS,
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean enableBzlmod;

    @Deprecated
    @Option(
        name = "enable_workspace",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = OptionEffectTag.LOADING_AND_ANALYSIS,
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean enableWorkspace;

    @Deprecated
    @Option(
        name = "experimental_announce_profile_path",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean announceProfilePath;

    @Deprecated
    @Option(
        name = "incompatible_existing_rules_immutable_view",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean incompatibleExistingRulesImmutableView;

    @Deprecated
    @Option(
        name = "incompatible_disable_native_repo_rules",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean incompatibleDisableNativeRepoRules;

    // Safe to delete after July 2025
    @Deprecated
    @Option(
        name = "incompatible_no_package_distribs",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean incompatibleNoPackageDistribs;

    @Deprecated
    @Option(
        name = "experimental_action_resource_set",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean experimentalActionResourceSet;

    @Deprecated
    @Option(
        name = "incompatible_macos_set_install_name",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean macosSetInstallName;

    @Deprecated
    @Option(
        name = "verbose_explanations",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean verboseExplanations;

    @Deprecated
    @Option(
        name = "experimental_cc_static_library",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean experimentalCcStaticLibrary;

    @Deprecated
    @Option(
        name = "incompatible_legacy_local_fallback",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public boolean legacyLocalFallback;
  }

  @Override
  public void initializeRuleClasses(ConfiguredRuleClassProvider.Builder builder) {
    BazelRuleClassProvider.setup(builder);
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(String commandName) {
    return commandName.equals("build")
        ? ImmutableList.of(BazelBuildGraveyardOptions.class, AllCommandGraveyardOptions.class)
        : ImmutableList.of(AllCommandGraveyardOptions.class);
  }

  @Override
  public void registerActionContexts(
      ModuleActionContextRegistry.Builder registryBuilder,
      CommandEnvironment env,
      BuildRequest buildRequest) {
    registryBuilder.register(JavaCompileActionContext.class, new JavaCompileActionContext());
  }
}
