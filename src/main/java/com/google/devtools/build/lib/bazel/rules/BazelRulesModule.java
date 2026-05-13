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
import com.google.devtools.common.options.OptionsClass;
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
  @OptionsClass
  public abstract static class BuildGraveyardOptions extends OptionsBase {

    @Deprecated
    @Option(
        name = "incompatible_enable_apple_toolchain_resolution",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public abstract boolean getIncompatibleUseToolchainResolution();

    @Deprecated
    @Option(
        name = "experimental_objc_provider_from_linked",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public abstract boolean getObjcProviderFromLinked();

    @Deprecated
    @Option(
        name = "build_python_zip",
        defaultValue = "auto",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        deprecationWarning =
            "The '--no' prefix is no longer supported for this flag. Please use"
                + " --build_python_zip=false instead.",
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public abstract TriState getBuildPythonZip();

    @Deprecated
    @Option(
        name = "incompatible_default_to_explicit_init_py",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public abstract boolean getIncompatibleDefaultToExplicitInitPy();

    @Deprecated
    @Option(
        name = "python_native_rules_allowlist",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        converter = LabelConverter.class,
        help = "Deprecated. No-op.")
    public abstract Label getNativeRulesAllowlist();

    @Deprecated
    @Option(
        name = "incompatible_python_disallow_native_rules",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public abstract boolean getDisallowNativeRules();

    @Deprecated
    @Option(
        name = "incompatible_remove_ctx_py_fragment",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public abstract boolean getDisablePyFragment();

    @Deprecated
    @Option(
        name = "incompatible_use_python_toolchains",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public abstract boolean getIncompatibleUsePythonToolchains();

    @Deprecated
    @Option(
        name = "experimental_starlark_cc_import",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public abstract boolean getExperimentalStarlarkCcImport();

    @Deprecated
    @Option(
        name = "experimental_platform_cc_test",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public abstract boolean getExperimentalPlatformCcTest();

    @Deprecated
    @Option(
        name = "j2objc_dead_code_removal",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public abstract boolean getRemoveDeadCode();

    @Deprecated
    @Option(
        name = "j2objc_translation_flags",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        allowMultiple = true,
        help = "Deprecated. No-op.")
    public abstract List<String> getTranslationFlags();

    @Deprecated
    @Option(
        name = "j2objc_dead_code_report",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public abstract String getDeadCodeReport();

    @Deprecated
    @Option(
        name = "experimental_proto_extra_actions",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.EXPERIMENTAL, OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public abstract boolean getExperimentalProtoExtraActions();

    @Deprecated
    @Option(
        name = "enable_fdo_profile_absolute_path",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public abstract boolean getEnableFdoProfileAbsolutePath();

    @Deprecated
    @Option(
        name = "experimental_genquery_use_graphless_query",
        defaultValue = "auto",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public abstract TriState getUseGraphlessQuery();

    @Deprecated
    @Option(
        name = "use_top_level_targets_for_symlinks",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public abstract boolean getUseTopLevelTargetsForSymlinks();

    @Deprecated
    @Option(
        name = "experimental_skyframe_prepare_analysis",
        deprecationWarning = "This flag is a no-op and will be deleted in a future release.",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public abstract boolean getSkyframePrepareAnalysis();

    @Deprecated
    @Option(
        name = "incompatible_disable_legacy_flags_cc_toolchain_api",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public abstract boolean getDisableLegacyFlagsCcToolchainApi();

    @Deprecated
    @Option(
        name = "incompatible_enable_profile_by_default",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getEnableProfileByDefault();

    @Deprecated
    @Option(
        name = "incompatible_override_toolchain_transition",
        defaultValue = "true",
        deprecationWarning = "This is now always set, please remove this flag.",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = OptionEffectTag.NO_OP,
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated, this is no longer in use and should be removed.")
    public abstract boolean getOverrideToolchainTransition();

    @Deprecated
    @Option(
        name = "experimental_parse_headers_skipped_if_corresponding_srcs_found",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getParseHeadersSkippedIfCorrespondingSrcsFound();

    @Deprecated
    @Option(
        name = "experimental_worker_as_resource",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op, will be removed soon.")
    public abstract boolean getWorkerAsResource();

    @Deprecated
    @Option(
        name = "high_priority_workers",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op, will be removed soon.",
        allowMultiple = true)
    public abstract List<String> getHighPriorityWorkers();

    @Deprecated
    @Option(
        name = "target_platform_fallback",
        defaultValue = "",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "This option is deprecated and has no effect.")
    public abstract String getTargetPlatformFallback();

    @Deprecated
    @Option(
        name = "incompatible_auto_configure_host_platform",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "This option is deprecated and has no effect.")
    public abstract boolean getAutoConfigureHostPlatform();

    @Deprecated
    @Option(
        name = "experimental_require_availability_info",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        defaultValue = "false",
        help = "Deprecated no-op.")
    public abstract boolean getRequireAvailabilityInfo();

    @Deprecated
    @Option(
        name = "experimental_collect_local_action_metrics",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated no-op.")
    public abstract boolean getCollectLocalExecutionStatistics();

    @Deprecated
    @Option(
        name = "experimental_collect_local_sandbox_action_metrics",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated no-op.")
    public abstract boolean getCollectLocalSandboxExecutionStatistics();

    @Deprecated
    @Option(
        name = "experimental_enable_starlark_doc_extract",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.EXPERIMENTAL, OptionMetadataTag.DEPRECATED},
        help = "Deprecated no-op.")
    public abstract boolean getEnableBzlDocDump();

    // TODO(b/274595070): Remove this option.
    @Deprecated
    @Option(
        name = "experimental_parallel_aquery_output",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getParallelAqueryOutput();

    @Deprecated
    @Option(
        name = "experimental_show_artifacts",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated no-op.")
    public abstract boolean getShowArtifacts();

    @Deprecated
    @Option(
        name = "announce",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.",
        deprecationWarning = "This option is now deprecated and is a no-op")
    public abstract boolean getAnnounce();

    @Deprecated
    @Option(
        name = "action_cache_store_output_metadata",
        oldName = "experimental_action_cache_store_output_metadata",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "no-op")
    public abstract boolean getActionCacheStoreOutputMetadata();

    @Deprecated
    @Option(
        name = "discard_actions_after_execution",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        effectTags = {OptionEffectTag.NO_OP},
        help = "This option is deprecated and has no effect.")
    public abstract boolean getDiscardActionsAfterExecution();

    @Deprecated
    @Option(
        name = "defer_param_files",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "This option is deprecated and has no effect and will be removed in the future.")
    public abstract boolean getDeferParamFiles();

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
    public abstract boolean getCheckFilesetDependenciesRecursively();

    @Deprecated
    @Option(
        name = "experimental_skyframe_native_filesets",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        deprecationWarning = "This flag is a no-op and skyframe-native-filesets is always true.")
    public abstract boolean getSkyframeNativeFileset();

    @Deprecated
    @Option(
        name = "collapse_duplicate_defines",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "no-op")
    public abstract boolean getCollapseDuplicateDefines();

    @Deprecated
    @Option(
        name = "incompatible_require_javaplugininfo_in_javacommon",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public abstract boolean getRequireJavaPluginInfo();

    @Deprecated
    @Option(
        name = "experimental_build_transitive_python_runfiles",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op",
        oldName = "incompatible_build_transitive_python_runfiles")
    public abstract boolean getBuildTransitiveRunfilesTrees();

    @Deprecated
    @Option(
        name = "experimental_use_new_worker_pool",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public abstract boolean getUseNewWorkerPool();

    @Deprecated
    @Option(
        name = "host_crosstool_top",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract String getHostCrosstoolTop();

    @Deprecated
    @Option(
        name = "experimental_use_semaphore_for_jobs",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getUseSemaphoreForJobs();

    // TODO(b/410585542): Remove this once there are no more internal users trying to set it.
    @Deprecated
    @Option(
        name = "experimental_skip_ttvs_for_genquery",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op. Will be removed soon.")
    public abstract boolean getSkipTtvs();

    @Deprecated
    @Option(
        name = "experimental_remote_analysis_cache",
        defaultValue = "",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract String getRemoteAnalysisCache();

    @Deprecated
    @Option(
        name = "experimental_remote_analysis_unreachable_cache_retry_interval",
        defaultValue = "",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public abstract String getUnreachableCacheRetryInterval();
  }

  /** This is where deprecated Bazel-specific options only used by the build command go to die. */
  @OptionsClass
  public abstract static class BazelBuildGraveyardOptions extends BuildGraveyardOptions {
    @Option(
        name = "python_path",
        defaultValue = "",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        // TODO: https://github.com/bazelbuild/bazel/pull/26854 - properly mark this deprecated.
        // This requires fixing the warning from
        // https://github.com/bazelbuild/bazel/blob/2576242fa867d0441b9015b004b7b35b6ed9187f/src/test/py/bazel/test_base.py#L120 on
        // Windows CI, which breaks
        // https://github.com/bazelbuild/bazel/blob/2576242fa867d0441b9015b004b7b35b6ed9187f/src/test/py/bazel/bzlmod/repo_contents_cache_test.py#L364.
        metadataTags = {OptionMetadataTag.HIDDEN},
        help = "Deprecated. No-op.")
    public abstract String getPythonPath();

    @Deprecated
    @Option(
        name = "experimental_python_import_all_repositories",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public abstract boolean getExperimentalPythonImportAllRepositories();

    @Deprecated
    @Option(
        name = "python_top",
        defaultValue = "",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public abstract String getPythonTop();

    @Deprecated
    @Option(
        name = "incompatible_load_python_rules_from_bzl",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "Deprecated no-op.")
    public abstract boolean getLoadPythonRulesFromBzl();

    @Deprecated
    @Option(
        name = "incompatible_load_proto_rules_from_bzl",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "Deprecated no-op.")
    public abstract boolean getLoadProtoRulesFromBzl();

    @Deprecated
    @Option(
        name = "incompatible_load_java_rules_from_bzl",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "Deprecated no-op.")
    public abstract boolean getLoadJavaRulesFromBzl();

    @Deprecated
    @Option(
        name = "make_variables_source",
        defaultValue = "configuration",
        metadataTags = {OptionMetadataTag.HIDDEN, OptionMetadataTag.DEPRECATED},
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP})
    public abstract String getMakeVariableSource();

    @Deprecated
    @Option(
        name = "force_ignore_dash_static",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "noop")
    public abstract boolean getForceIgnoreDashStatic();

    @Deprecated
    @Option(
        name = "experimental_profile_action_counts",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getEnableActionCountProfile();

    @Deprecated
    @Option(
        name = "incompatible_remove_binary_profile",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getRemoveBinaryProfile();

    @Deprecated
    @Option(
        name = "experimental_post_profile_started_event",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getPostProfileStartedEvent();

    @Deprecated
    @Option(
        name = "incompatible_dont_use_javasourceinfoprovider",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public abstract boolean getDontUseJavaSourceInfoProvider();

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
    public abstract String getSdk();

    @Deprecated
    @Option(
        name = "android_cpu",
        defaultValue = "",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op",
        deprecationWarning = ANDROID_FLAG_DEPRECATION)
    public abstract String getCpu();

    @Deprecated
    @Option(
        name = "android_crosstool_top",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op",
        deprecationWarning = ANDROID_FLAG_DEPRECATION)
    public abstract String getAndroidCrosstoolTop();

    @Deprecated
    @Option(
        name = "android_grte_top",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op",
        deprecationWarning = ANDROID_FLAG_DEPRECATION)
    public abstract String getAndroidLibcTopLabel();

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
    public abstract List<String> getFatApkCpus();

    @Deprecated
    @Option(
        name = "incompatible_enable_cgo_toolchain_resolution",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public abstract boolean getIncompatibleEnableGoToolchainResolution();
  }

  /**
   * This is where deprecated options which need to be available for all commands go to die. If you
   * want to graveyard an all-command option specific to Blaze or Bazel, create a subclass.
   */
  @OptionsClass
  public abstract static class AllCommandGraveyardOptions extends OptionsBase {

    @Deprecated
    @Option(
        name = "incompatible_autoload_externally",
        converter = Converters.CommaSeparatedOptionSetConverter.class,
        defaultValue = "",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public abstract List<String> getIncompatibleAutoloadExternally();

    @Deprecated
    @Option(
        name = "repositories_without_autoloads",
        converter = Converters.CommaSeparatedOptionSetConverter.class,
        defaultValue = "",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public abstract List<String> getRepositoriesWithoutAutoloads();

    @Deprecated
    @Option(
        name = "incompatible_disable_autoloads_in_main_repo",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "Deprecated. No-op.")
    public abstract boolean getIncompatibleDisableAutoloadsInMainRepo();

    @Deprecated
    @Option(
        name = "experimental_py_binaries_include_label",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getIncludeLabelInPyBinariesLinkstamp();

    @Deprecated
    @Option(
        name = "incompatible_python_disable_py2",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getDisablePy2();

    @Deprecated
    @Option(
        name = "force_python",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract String getForcePython();

    @Deprecated
    @Option(
        name = "host_force_python",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract String getHostForcePython();

    @Deprecated
    @Option(
        name = "incompatible_py3_is_default",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getIncompatiblePy3IsDefault();

    @Deprecated
    @Option(
        name = "incompatible_py2_outputs_are_suffixed",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getIncompatiblePy2OutputsAreSuffixed();

    @Deprecated
    @Option(
        name = "python_version",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract String getPythonVersion();

    @Deprecated
    @Option(
        name = "incompatible_remove_old_python_version_api",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getIncompatibleRemoveOldPythonVersionApi();

    @Deprecated
    @Option(
        name = "incompatible_allow_python_version_transitions",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getIncompatibleAllowPythonVersionTransitions();

    @Deprecated
    @Option(
        name = "incompatible_disallow_legacy_py_provider",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getIncompatibleDisallowLegacyPyProvider();

    @Deprecated
    @Option(
        name = "experimental_disallow_legacy_java_toolchain_flags",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getExperimentalDisallowLegacyJavaToolchainFlags();

    @Deprecated
    @Option(
        name = "javabase",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract String getJavabase();

    @Deprecated
    @Option(
        name = "java_toolchain",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract String getJavaToolchain();

    @Deprecated
    @Option(
        name = "host_java_toolchain",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract String getHostJavaToolchain();

    @Deprecated
    @Option(
        name = "host_javabase",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract String getHostJavabase();

    @Deprecated
    @Option(
        name = "apple_crosstool_top",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public abstract String getAppleCrosstoolTop();

    @Deprecated
    @Option(
        name = "legacy_bazel_java_test",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public abstract boolean getLegacyBazelJavaTest();

    @Deprecated
    @Option(
        name = "strict_deps_java_protos",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public abstract boolean getStrictDepsJavaProtos();

    @Deprecated
    @Option(
        name = "disallow_strict_deps_for_jpl",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public abstract boolean getIsDisallowStrictDepsForJpl();

    @Deprecated
    @Option(
        name = "experimental_import_deps_checking",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public abstract String getImportDepsCheckingLevel();

    @Deprecated
    @Option(
        name = "experimental_allow_runtime_deps_on_neverlink",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public abstract boolean getAllowRuntimeDepsOnNeverLink();

    @Deprecated
    @Option(
        name = "experimental_limit_android_lint_to_android_constrained_java",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public abstract boolean getLimitAndroidLintToAndroidCompatible();

    @Deprecated
    @Option(
        name = "experimental_java_header_input_pruning",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public abstract boolean getExperimentalJavaHeaderInputPruning();

    @Deprecated
    @Option(
        name = "incompatible_dont_collect_native_libraries_in_data",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getDontCollectDataLibraries();

    @Deprecated
    @Option(
        name = "experimental_java_header_compilation_direct_deps",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getJavaHeaderCompilationDirectDeps();

    @Deprecated
    @Option(
        name = "jplPropagateCcLinkParamsStore",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public abstract boolean getJplPropagateCcLinkParamsStore();

    @Deprecated
    @Option(
        name = "incompatible_require_linker_input_cc_api",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public abstract boolean getIncompatibleRequireLinkerInputCcApi();

    @Deprecated
    @Option(
        name = "incompatible_depset_for_libraries_to_link_getter",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public abstract boolean getIncompatibleDepsetForLibrariesToLinkGetter();

    @Deprecated
    @Option(
        name = "legacy_external_runfiles",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public abstract boolean getLegacyExternalRunfiles();

    @Deprecated
    @Option(
        name = "incompatible_disable_target_provider_fields",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public abstract boolean getIncompatibleDisableTargetProviderFields();

    @Deprecated
    @Option(
        name = "incompatible_disallow_struct_provider_syntax",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public abstract boolean getIncompatibleDisallowStructProviderSyntax();

    @Deprecated
    @Option(
        name = "auto_cpu_environment_group",
        defaultValue = "",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {
          OptionMetadataTag.EXPERIMENTAL,
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.DEPRECATED
        },
        help = "No-op")
    public abstract String getAutoCpuEnvironmentGroup();

    @Deprecated
    @Option(
        name = "incompatible_top_level_aspects_require_providers",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        effectTags = {OptionEffectTag.NO_OP},
        help = "No-op")
    public abstract boolean getIncompatibleTopLevelAspectsRequireProviders();

    @Deprecated
    @Option(
        name = "separate_aspect_deps",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public abstract boolean getSeparateAspectDeps();

    @Deprecated
    @Option(
        name = "incompatible_visibility_private_attributes_at_definition",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "No-op")
    public abstract boolean getIncompatibleVisibilityPrivateAttributesAtDefinition();

    @Deprecated
    @Option(
        name = "incompatible_new_actions_api",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        effectTags = {OptionEffectTag.NO_OP},
        help = "No-op")
    public abstract boolean getIncompatibleNewActionsApi();

    @Deprecated
    @Option(
        name = "experimental_enable_aspect_hints",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
        metadataTags = {OptionMetadataTag.EXPERIMENTAL, OptionMetadataTag.DEPRECATED})
    public abstract boolean getEnableAspectHints();

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
    public abstract boolean getBesBestEffort();

    @Deprecated
    @Option(
        name = "experimental_use_event_based_build_completion_status",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        metadataTags = {OptionMetadataTag.EXPERIMENTAL, OptionMetadataTag.DEPRECATED},
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.EXECUTION},
        help = "No-op")
    public abstract boolean getUseEventBasedBuildCompletionStatus();

    @Deprecated
    @Option(
        name = "experimental_java_proto_add_allowed_public_imports",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.EXPERIMENTAL, OptionMetadataTag.DEPRECATED},
        help = "This flag is a noop and scheduled for removal.")
    public abstract boolean getExperimentalJavaProtoAddAllowedPublicImports();

    @Deprecated
    @Option(
        name = "java_optimization_mode",
        defaultValue = "",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Do not use.")
    public abstract String getJavaOptimizationMode();

    @Deprecated
    @Option(
        name = "incompatible_depset_for_java_output_source_jars",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getIncompatibleDepsetForJavaOutputSourceJars();

    @Deprecated
    @Option(
        name = "incompatible_use_plus_in_repo_names",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = OptionEffectTag.LOADING_AND_ANALYSIS,
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getIncompatibleUsePlusInRepoNames();

    @Deprecated
    @Option(
        name = "enable_bzlmod",
        oldName = "experimental_enable_bzlmod",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = OptionEffectTag.LOADING_AND_ANALYSIS,
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getEnableBzlmod();

    @Deprecated
    @Option(
        name = "enable_workspace",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = OptionEffectTag.LOADING_AND_ANALYSIS,
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getEnableWorkspace();

    @Deprecated
    @Option(
        name = "experimental_announce_profile_path",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getAnnounceProfilePath();

    @Deprecated
    @Option(
        name = "incompatible_existing_rules_immutable_view",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getIncompatibleExistingRulesImmutableView();

    @Deprecated
    @Option(
        name = "incompatible_disable_native_repo_rules",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getIncompatibleDisableNativeRepoRules();

    // Safe to delete after July 2025
    @Deprecated
    @Option(
        name = "incompatible_no_package_distribs",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getIncompatibleNoPackageDistribs();

    @Deprecated
    @Option(
        name = "experimental_action_resource_set",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getExperimentalActionResourceSet();

    @Deprecated
    @Option(
        name = "incompatible_macos_set_install_name",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
        metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getMacosSetInstallName();

    @Deprecated
    @Option(
        name = "verbose_explanations",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getVerboseExplanations();

    @Deprecated
    @Option(
        name = "experimental_cc_static_library",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getExperimentalCcStaticLibrary();

    @Deprecated
    @Option(
        name = "incompatible_legacy_local_fallback",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getLegacyLocalFallback();

    @Deprecated
    @Option(
        name = "proto_toolchain_for_j2objc",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        converter = LabelConverter.class,
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract Label getProtoToolchainForJ2Objc();

    @Deprecated
    @Option(
        name = "incompatible_use_cc_configure_from_rules_cc",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "No-op.")
    public abstract boolean getIncompatibleUseCcConfigureFromRulesCc();
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
