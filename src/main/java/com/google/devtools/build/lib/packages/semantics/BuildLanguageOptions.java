// Copyright 2019 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.packages.semantics;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.common.options.Converters.CommaSeparatedNonEmptyOptionListConverter;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;
import java.util.List;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * Options that affect the semantics of Bazel's build language.
 *
 * <p>These are injected into Skyframe (as an instance of {@link StarlarkSemantics}) when a new
 * build invocation occurs. Changing these options between builds will therefore trigger a
 * reevaluation of everything that depends on the Starlark interpreter &mdash; in particular,
 * evaluation of all BUILD and .bzl files.
 *
 * <p><em>To add a new option, update the following:</em>
 *
 * <ul>
 *   <li>Add a new {@code @Option}-annotated field to this class. The field name and default value
 *       should be the same as in {@link StarlarkSemantics}, and the option name in the annotation
 *       should be that name written in snake_case. Add a line to set the new field in {@link
 *       #toStarlarkSemantics}.
 *   <li>Define a new {@code StarlarkSemantics.Key} or {@code StarlarkSemantics} boolean flag
 *       identifier.
 *   <li>Add a line to set the new field in both {@link ConsistencyTest#buildRandomOptions} and
 *       {@link ConsistencyTest#buildRandomSemantics}.
 *   <li>Update manual documentation in site/docs/skylark/backward-compatibility.md. Also remember
 *       to update this when flipping a flag's default value.
 *   <li>Boolean semantic flags can toggle StarlarkMethod-annotated Java methods (or their
 *       parameters) on or off, making them selectively invisible to Starlark. To do this, add a new
 *       entry to {@link BuildLanguageOptions}, then specify the identifier in {@link
 *       StarlarkMethod#enableOnlyWithFlag} or {@link StarlarkMethod#disableWithFlag}.
 * </ul>
 */
public final class BuildLanguageOptions extends OptionsBase {
  @Option(
      name = "experimental_java_library_export",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help = "If enabled, experimental_java_library_export_do_not_use module is available.")
  public boolean experimentalJavaLibraryExport;

  @Option(
      name = "incompatible_stop_exporting_language_modules",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "If enabled, certain language-specific modules (such as `cc_common`) are unavailable in"
              + " user .bzl files and may only be called from their respective rules repositories.")
  public boolean incompatibleStopExportingLanguageModules;

  @Option(
      name = "incompatible_remove_rule_name_parameter",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.NO_OP},
      metadataTags = {OptionMetadataTag.DEPRECATED},
      help = "No-op")
  public boolean incompatibleRemoveRuleNameParameter;

  @Option(
      name = "incompatible_disallow_symlink_file_to_dir",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help = "No-op.")
  public boolean incompatibleDisallowSymlinkFileToDir;

  @Option(
      name = "experimental_build_setting_api",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = OptionEffectTag.BUILD_FILE_SEMANTICS,
      help =
          "If set to true, allows access to value of build setting rules via "
              + "ctx.build_setting_value.")
  public boolean experimentalBuildSettingApi;

  // TODO(#11437): Delete the special empty string value so that it's on unconditionally.
  @Option(
      name = "experimental_builtins_bzl_path",
      defaultValue = "%bundled%",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOSES_INCREMENTAL_STATE, OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "This flag tells Bazel how to find the \"@_builtins\" .bzl files that govern how "
              + "predeclared symbols for BUILD and .bzl files are defined. This flag is only "
              + "intended for Bazel developers, to help when writing @_builtins .bzl code. "
              + "Ordinarily this value is set to \"%bundled%\", which means to use the "
              + "builtins_bzl/ directory packaged in the Bazel binary. However, it can be set to "
              + "the path (relative to the root of the current workspace) of an alternate "
              + "builtins_bzl/ directory, such as one in a Bazel source tree workspace. A literal "
              + "value of \"%workspace%\" is equivalent to the relative package path of "
              + "builtins_bzl/ within a Bazel source tree; this should only be used when running "
              + "Bazel within its own source tree. Finally, a value of the empty string disables "
              + "the builtins injection mechanism entirely.")
  public String experimentalBuiltinsBzlPath;

  @Option(
      name = "experimental_builtins_dummy",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help = "Enables an internal dummy symbol used to test builtins injection.")
  public boolean experimentalBuiltinsDummy;

  @Option(
      name = "experimental_builtins_injection_override",
      converter = CommaSeparatedNonEmptyOptionListConverter.class,
      defaultValue = "null",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "A comma-separated list of symbol names prefixed by a + or - character, indicating which"
              + " symbols from `@_builtins//:exports.bzl` to inject, overriding their default"
              + " injection status. Precisely, this works as follows. Each dict key of"
              + " `exported_toplevels` or `exported_rules` has the form `foo`, `+foo`, or `-foo`."
              + " The first two forms mean it gets injected by default, while the last form means"
              + " it does not get injected by default. In the first case (unprefixed), the default"
              + " is absolute and cannot be overridden. Otherwise, we then consult this options"
              + " list, and if we see foo occur here, we take the prefix of its last occurrence and"
              + " use that to decide whether or not to inject. It is a no-op to specify an unknown"
              + " symbol, or to attempt to not inject a symbol that occurs unprefixed in a dict"
              + " key.")
  public List<String> experimentalBuiltinsInjectionOverride;

  @Option(
      name = "experimental_bzl_visibility",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "If enabled, adds a `visibility()` function that .bzl files may call during top-level"
              + " evaluation to set their visibility for the purpose of load() statements.")
  public boolean experimentalBzlVisibility;

  @Option(
      name = "check_bzl_visibility",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      help = "If disabled, .bzl load visibility errors are demoted to warnings.")
  public boolean checkBzlVisibility;

  @Option(
      name = "experimental_cc_skylark_api_enabled_packages",
      converter = CommaSeparatedOptionListConverter.class,
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "Passes list of packages that can use the C++ Starlark API. Don't enable this flag yet, "
              + "we will be making breaking changes.")
  public List<String> experimentalCcStarlarkApiEnabledPackages;

  @Option(
      name = "experimental_enable_android_migration_apis",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = OptionEffectTag.BUILD_FILE_SEMANTICS,
      help = "If set to true, enables the APIs required to support the Android Starlark migration.")
  public boolean experimentalEnableAndroidMigrationApis;

  @Option(
      name = "experimental_enable_scl_dialect",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = OptionEffectTag.BUILD_FILE_SEMANTICS,
      // TODO(brandjon): point to more extensive user documentation somewhere
      help = "If set to true, .scl files may be used in load() statements.")
  public boolean experimentalEnableSclDialect;

  @Option(
      name = "enable_bzlmod",
      oldName = "experimental_enable_bzlmod",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = OptionEffectTag.LOADING_AND_ANALYSIS,
      help =
          "If true, enables the Bzlmod dependency management system, taking precedence over"
              + " WORKSPACE. See https://bazel.build/docs/bzlmod for more information.")
  public boolean enableBzlmod;

  @Option(
      name = "experimental_isolated_extension_usages",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = OptionEffectTag.LOADING_AND_ANALYSIS,
      help =
          "If true, enables the <code>isolate</code> parameter in the <a"
              + " href=\"https://bazel.build/rules/lib/globals/module#use_extension\"><code>use_extension</code></a>"
              + " function.")
  public boolean experimentalIsolatedExtensionUsages;

  @Option(
      name = "experimental_java_proto_library_default_has_services",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = OptionEffectTag.BUILD_FILE_SEMANTICS,
      help = "The default value of java_proto_library.has_services.")
  public boolean experimentalJavaProtoLibraryDefaultHasServices;

  @Option(
      name = "incompatible_existing_rules_immutable_view",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS, OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "If set to true, native.existing_rule and native.existing_rules return lightweight"
              + " immutable view objects instead of mutable dicts.")
  public boolean incompatibleExistingRulesImmutableView;

  @Option(
      name = "experimental_google_legacy_api",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "If set to true, exposes a number of experimental pieces of Starlark build API "
              + "pertaining to Google legacy code.")
  public boolean experimentalGoogleLegacyApi;

  @Option(
      name = "experimental_platforms_api",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "If set to true, enables a number of platform-related Starlark APIs useful for "
              + "debugging.")
  public boolean experimentalPlatformsApi;

  @Option(
      name = "experimental_cc_shared_library",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS, OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {
        OptionMetadataTag.EXPERIMENTAL,
      },
      help =
          "If set to true, rule attributes and Starlark API methods needed for the rule "
              + "cc_shared_library will be available")
  public boolean experimentalCcSharedLibrary;

  @Option(
      name = "incompatible_require_linker_input_cc_api",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS, OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "If set to true, rule create_linking_context will require linker_inputs instead of "
              + "libraries_to_link. The old getters of linking_context will also be disabled and "
              + "just linker_inputs will be available.")
  public boolean incompatibleRequireLinkerInputCcApi;

  @Option(
      name = "experimental_repo_remote_exec",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS, OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {
        OptionMetadataTag.EXPERIMENTAL,
      },
      help = "If set to true, repository_rule gains some remote execution capabilities.")
  public boolean experimentalRepoRemoteExec;

  @Option(
      name = "experimental_disable_external_package",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.LOSES_INCREMENTAL_STATE},
      metadataTags = {
        OptionMetadataTag.EXPERIMENTAL,
      },
      help =
          "If set to true, the auto-generated //external package will not be available anymore. "
              + "Bazel will still be unable to parse the file 'external/BUILD', but globs reaching "
              + "into external/ from the unnamed package will work.")
  public boolean experimentalDisableExternalPackage;

  @Option(
      name = "experimental_sibling_repository_layout",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {
        OptionEffectTag.ACTION_COMMAND_LINES,
        OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION,
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.LOSES_INCREMENTAL_STATE
      },
      metadataTags = {
        OptionMetadataTag.EXPERIMENTAL,
      },
      help =
          "If set to true, non-main repositories are planted as symlinks to the main repository in"
              + " the execution root. That is, all repositories are direct children of the"
              + " $output_base/execution_root directory. This has the side effect of freeing up"
              + " $output_base/execution_root/__main__/external for the real top-level 'external' "
              + "directory.")
  public boolean experimentalSiblingRepositoryLayout;

  @Option(
      name = "incompatible_allow_tags_propagation",
      oldName = "experimental_allow_tags_propagation",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.EXPERIMENTAL,
      },
      help =
          "If set to true, tags will be propagated from a target to the actions' execution"
              + " requirements; otherwise tags are not propagated. See"
              + " https://github.com/bazelbuild/bazel/issues/8830 for details.")
  public boolean experimentalAllowTagsPropagation;

  @Option(
      name = "experimental_action_resource_set",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.EXECUTION, OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.EXPERIMENTAL,
      },
      help =
          "If set to true, ctx.actions.run() and ctx.actions.run_shell() accept a resource_set"
              + " parameter for local execution. Otherwise it will default to 250 MB for memory"
              + " and 1 cpu.")
  public boolean experimentalActionResourceSet;

  @Option(
      name = "experimental_lazy_template_expansion",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.EXECUTION, OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.EXPERIMENTAL,
      },
      help =
          "If set to true, ctx.actions.expand_template() accepts a TemplateDict"
              + " parameter for deferred evaluation of substitution values.")
  public boolean experimentalLazyTemplateExpansion;

  @Option(
      name = "experimental_analysis_test_call",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.EXPERIMENTAL,
      },
      help = "If set to true, analysis_test native call is available.")
  public boolean experimentalAnalysisTestCall;

  @Option(
      name = "incompatible_struct_has_no_methods",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "Disables the to_json and to_proto methods of struct, which pollute the struct field"
              + " namespace. Instead, use json.encode or json.encode_indent for JSON, or"
              + " proto.encode_text for textproto.")
  public boolean incompatibleStructHasNoMethods;

  @Option(
      name = "incompatible_always_check_depset_elements",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "Check the validity of elements added to depsets, in all constructors. Elements must be"
              + " immutable, but historically the depset(direct=...) constructor forgot to check."
              + " Use tuples instead of lists in depset elements."
              + " See https://github.com/bazelbuild/bazel/issues/10313 for details.")
  public boolean incompatibleAlwaysCheckDepsetElements;

  @Option(
      name = "incompatible_disable_target_provider_fields",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "If set to true, disable the ability to access providers on 'target' objects via field "
              + "syntax. Use provider-key syntax instead. For example, instead of using "
              + "`ctx.attr.dep.my_info` to access `my_info` from inside a rule implementation "
              + "function, use `ctx.attr.dep[MyInfo]`. See "
              + "https://github.com/bazelbuild/bazel/issues/9014 for details.")
  public boolean incompatibleDisableTargetProviderFields;

  @Option(
      name = "incompatible_disallow_empty_glob",
      defaultValue = "false",
      category = "incompatible changes",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help = "If set to true, the default value of the `allow_empty` argument of glob() is False.")
  public boolean incompatibleDisallowEmptyGlob;

  @Option(
      name = "incompatible_disallow_legacy_javainfo",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help = "Deprecated. No-op.")
  // TODO(elenairina): Move option to graveyard after the flag is removed from the global blazerc.
  public boolean incompatibleDisallowLegacyJavaInfo;

  @Option(
      name = "incompatible_disallow_struct_provider_syntax",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "If set to true, rule implementation functions may not return a struct. They must "
              + "instead return a list of provider instances.")
  public boolean incompatibleDisallowStructProviderSyntax;

  @Option(
      name = "incompatible_package_group_has_public_syntax",
      defaultValue = FlagConstants.DEFAULT_INCOMPATIBLE_PACKAGE_GROUP_HAS_PUBLIC_SYNTAX,
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "In package_group's `packages` attribute, allows writing \"public\" or \"private\" to"
              + " refer to all packages or no packages respectively.")
  public boolean incompatiblePackageGroupHasPublicSyntax;

  @Option(
      name = "incompatible_fix_package_group_reporoot_syntax",
      defaultValue = FlagConstants.DEFAULT_INCOMPATIBLE_FIX_PACKAGE_GROUP_REPOROOT_SYNTAX,
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "In package_group's `packages` attribute, changes the meaning of the value \"//...\" to"
              + " refer to all packages in the current repository instead of all packages in any"
              + " repository. You can use the special value \"public\" in place of \"//...\" to"
              + " obtain the old behavior. This flag requires"
              + " that --incompatible_package_group_has_public_syntax also be enabled.")
  public boolean incompatibleFixPackageGroupReporootSyntax;

  @Option(
      name = "incompatible_visibility_private_attributes_at_definition",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "If set to true, the visibility of private rule attributes is checked with respect "
              + "to the rule definition, rather than the rule usage.")
  public boolean incompatibleVisibilityPrivateAttributesAtDefinition;

  @Option(
      name = "incompatible_new_actions_api",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "If set to true, the API to create actions is only available on `ctx.actions`, "
              + "not on `ctx`.")
  public boolean incompatibleNewActionsApi;

  @Option(
      name = "incompatible_no_attr_license",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help = "If set to true, disables the function `attr.license`.")
  public boolean incompatibleNoAttrLicense;

  @Option(
      name = "incompatible_no_implicit_file_export",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "If set, (used) source files are are package private unless exported explicitly. See "
              + "https://github.com/bazelbuild/proposals/blob/master/designs/"
              + "2019-10-24-file-visibility.md")
  public boolean incompatibleNoImplicitFileExport;

  @Option(
      name = "incompatible_no_rule_outputs_param",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help = "If set to true, disables the `outputs` parameter of the `rule()` Starlark function.")
  public boolean incompatibleNoRuleOutputsParam;

  @Option(
      name = "incompatible_run_shell_command_string",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help = "If set to true, the command parameter of actions.run_shell will only accept string")
  public boolean incompatibleRunShellCommandString;

  /** Used in an integration test to confirm that flags are visible to the interpreter. */
  @Option(
      name = "internal_starlark_flag_test_canary",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN})
  public boolean internalStarlarkFlagTestCanary;

  @Option(
      name = "incompatible_do_not_split_linking_cmdline",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "When true, Bazel no longer modifies command line flags used for linking, and also "
              + "doesn't selectively decide which flags go to the param file and which don't.  "
              + "See https://github.com/bazelbuild/bazel/issues/7670 for details.")
  public boolean incompatibleDoNotSplitLinkingCmdline;

  @Option(
      name = "incompatible_use_cc_configure_from_rules_cc",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "When true, Bazel will no longer allow using cc_configure from @bazel_tools. "
              + "Please see https://github.com/bazelbuild/bazel/issues/10134 for details and "
              + "migration instructions.")
  public boolean incompatibleUseCcConfigureFromRulesCc;

  @Option(
      name = "incompatible_unambiguous_label_stringification",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "When true, Bazel will stringify the label @//foo:bar to @//foo:bar, instead of"
              + " //foo:bar. This only affects the behavior of str(), the % operator, and so on;"
              + " the behavior of repr() is unchanged. See"
              + " https://github.com/bazelbuild/bazel/issues/15916 for more information.")
  public boolean incompatibleUnambiguousLabelStringification;

  @Option(
      name = "incompatible_depset_for_libraries_to_link_getter",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "When true, Bazel no longer returns a list from linking_context.libraries_to_link but "
              + "returns a depset instead.")
  public boolean incompatibleDepsetForLibrariesToLinkGetter;

  @Option(
      name = "incompatible_depset_for_java_output_source_jars",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "When true, Bazel no longer returns a list from java_info.java_output[0].source_jars but "
              + "returns a depset instead.")
  public boolean incompatibleDepsetForJavaOutputSourceJars;

  @Option(
      name = "incompatible_java_common_parameters",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "If set to true, the output_jar, and host_javabase parameters in pack_sources and "
              + "host_javabase in compile will all be removed.")
  public boolean incompatibleJavaCommonParameters;

  @Option(
      name = "max_computation_steps",
      defaultValue = "0",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      help =
          "The maximum number of Starlark computation steps that may be executed by a BUILD file"
              + " (zero means no limit).")
  public long maxComputationSteps;

  @Option(
      name = "nested_set_depth_limit",
      defaultValue = "3500",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      help =
          "The maximum depth of the graph internal to a depset (also known as NestedSet), above"
              + " which the depset() constructor will fail.")
  public int nestedSetDepthLimit;

  @Option(
      name = "incompatible_top_level_aspects_require_providers",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      help =
          "If set to true, the top level aspect will honor its required providers and only run on"
              + " top level targets whose rules' advertised providers satisfy the required"
              + " providers of the aspect.")
  public boolean incompatibleTopLevelAspectsRequireProviders;

  @Option(
      name = "incompatible_disable_starlark_host_transitions",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      help =
          "If set to true, rule attributes cannot set 'cfg = \"host\"'. Rules should set "
              + "'cfg = \"exec\"' instead.")
  public boolean incompatibleDisableStarlarkHostTransitions;

  @Option(
      name = "experimental_get_fixed_configured_action_env",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "If enabled, action.env will also return fixed environment variables"
              + " specified through features configuration.")
  public boolean experimentalGetFixedConfiguredEnvironment;

  /**
   * An interner to reduce the number of StarlarkSemantics instances. A single Blaze instance should
   * never accumulate a large number of these and being able to shortcut on object identity makes a
   * comparison later much faster. In particular, the semantics become part of the
   * MethodDescriptorKey in CallExpression and are thus compared for every function call.
   */
  private static final Interner<StarlarkSemantics> INTERNER = BlazeInterners.newWeakInterner();

  /** Constructs a {@link StarlarkSemantics} object corresponding to this set of option values. */
  public StarlarkSemantics toStarlarkSemantics() {
    // This function connects command-line flags to their corresponding StarlarkSemantics keys.
    StarlarkSemantics semantics =
        StarlarkSemantics.builder()
            .setBool(EXPERIMENTAL_JAVA_LIBRARY_EXPORT, experimentalJavaLibraryExport)
            .setBool(
                INCOMPATIBLE_STOP_EXPORTING_LANGUAGE_MODULES,
                incompatibleStopExportingLanguageModules)
            .setBool(INCOMPATIBLE_ALLOW_TAGS_PROPAGATION, experimentalAllowTagsPropagation)
            .set(EXPERIMENTAL_BUILTINS_BZL_PATH, experimentalBuiltinsBzlPath)
            .setBool(EXPERIMENTAL_BUILTINS_DUMMY, experimentalBuiltinsDummy)
            .set(EXPERIMENTAL_BUILTINS_INJECTION_OVERRIDE, experimentalBuiltinsInjectionOverride)
            .setBool(EXPERIMENTAL_BZL_VISIBILITY, experimentalBzlVisibility)
            .setBool(CHECK_BZL_VISIBILITY, checkBzlVisibility)
            .setBool(
                EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS, experimentalEnableAndroidMigrationApis)
            .setBool(EXPERIMENTAL_ENABLE_SCL_DIALECT, experimentalEnableSclDialect)
            .setBool(ENABLE_BZLMOD, enableBzlmod)
            .setBool(EXPERIMENTAL_ISOLATED_EXTENSION_USAGES, experimentalIsolatedExtensionUsages)
            .setBool(
                EXPERIMENTAL_JAVA_PROTO_LIBRARY_DEFAULT_HAS_SERVICES,
                experimentalJavaProtoLibraryDefaultHasServices)
            .setBool(
                INCOMPATIBLE_EXISTING_RULES_IMMUTABLE_VIEW, incompatibleExistingRulesImmutableView)
            .setBool(EXPERIMENTAL_ACTION_RESOURCE_SET, experimentalActionResourceSet)
            .setBool(EXPERIMENTAL_LAZY_TEMPLATE_EXPANSION, experimentalLazyTemplateExpansion)
            .setBool(EXPERIMENTAL_ANALYSIS_TEST_CALL, experimentalAnalysisTestCall)
            .setBool(EXPERIMENTAL_GOOGLE_LEGACY_API, experimentalGoogleLegacyApi)
            .setBool(EXPERIMENTAL_PLATFORMS_API, experimentalPlatformsApi)
            .setBool(EXPERIMENTAL_CC_SHARED_LIBRARY, experimentalCcSharedLibrary)
            .setBool(EXPERIMENTAL_REPO_REMOTE_EXEC, experimentalRepoRemoteExec)
            .setBool(EXPERIMENTAL_DISABLE_EXTERNAL_PACKAGE, experimentalDisableExternalPackage)
            .setBool(EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT, experimentalSiblingRepositoryLayout)
            .setBool(
                INCOMPATIBLE_DISABLE_TARGET_PROVIDER_FIELDS,
                incompatibleDisableTargetProviderFields)
            .setBool(
                INCOMPATIBLE_ALWAYS_CHECK_DEPSET_ELEMENTS, incompatibleAlwaysCheckDepsetElements)
            .setBool(INCOMPATIBLE_DISALLOW_EMPTY_GLOB, incompatibleDisallowEmptyGlob)
            .setBool(
                INCOMPATIBLE_DISALLOW_STRUCT_PROVIDER_SYNTAX,
                incompatibleDisallowStructProviderSyntax)
            .setBool(
                INCOMPATIBLE_PACKAGE_GROUP_HAS_PUBLIC_SYNTAX,
                incompatiblePackageGroupHasPublicSyntax)
            .setBool(
                INCOMPATIBLE_FIX_PACKAGE_GROUP_REPOROOT_SYNTAX,
                incompatibleFixPackageGroupReporootSyntax)
            .setBool(INCOMPATIBLE_JAVA_COMMON_PARAMETERS, incompatibleJavaCommonParameters)
            .setBool(INCOMPATIBLE_NEW_ACTIONS_API, incompatibleNewActionsApi)
            .setBool(INCOMPATIBLE_NO_ATTR_LICENSE, incompatibleNoAttrLicense)
            .setBool(INCOMPATIBLE_NO_IMPLICIT_FILE_EXPORT, incompatibleNoImplicitFileExport)
            .setBool(INCOMPATIBLE_NO_RULE_OUTPUTS_PARAM, incompatibleNoRuleOutputsParam)
            .setBool(INCOMPATIBLE_RUN_SHELL_COMMAND_STRING, incompatibleRunShellCommandString)
            .setBool(INCOMPATIBLE_STRUCT_HAS_NO_METHODS, incompatibleStructHasNoMethods)
            .setBool(
                INCOMPATIBLE_VISIBILITY_PRIVATE_ATTRIBUTES_AT_DEFINITION,
                incompatibleVisibilityPrivateAttributesAtDefinition)
            .setBool(StarlarkSemantics.PRINT_TEST_MARKER, internalStarlarkFlagTestCanary)
            .setBool(
                INCOMPATIBLE_DO_NOT_SPLIT_LINKING_CMDLINE, incompatibleDoNotSplitLinkingCmdline)
            .setBool(
                INCOMPATIBLE_USE_CC_CONFIGURE_FROM_RULES_CC, incompatibleUseCcConfigureFromRulesCc)
            .setBool(
                INCOMPATIBLE_UNAMBIGUOUS_LABEL_STRINGIFICATION,
                incompatibleUnambiguousLabelStringification)
            .setBool(
                INCOMPATIBLE_DEPSET_FOR_LIBRARIES_TO_LINK_GETTER,
                incompatibleDepsetForLibrariesToLinkGetter)
            .setBool(
                INCOMPATIBLE_DEPSET_FOR_JAVA_OUTPUT_SOURCE_JARS,
                incompatibleDepsetForJavaOutputSourceJars)
            .setBool(INCOMPATIBLE_REQUIRE_LINKER_INPUT_CC_API, incompatibleRequireLinkerInputCcApi)
            .set(MAX_COMPUTATION_STEPS, maxComputationSteps)
            .set(NESTED_SET_DEPTH_LIMIT, nestedSetDepthLimit)
            .setBool(
                INCOMPATIBLE_TOP_LEVEL_ASPECTS_REQUIRE_PROVIDERS,
                incompatibleTopLevelAspectsRequireProviders)
            .setBool(
                INCOMPATIBLE_DISABLE_STARLARK_HOST_TRANSITIONS,
                incompatibleDisableStarlarkHostTransitions)
            .setBool(
                EXPERIMENTAL_GET_FIXED_CONFIGURED_ACTION_ENV,
                experimentalGetFixedConfiguredEnvironment)
            .build();
    return INTERNER.intern(semantics);
  }

  // StarlarkSemantics keys used by Bazel
  //
  // Sadly it is impossible to move most of these declarations closer to the
  // code they affect: the toStarlarkSemantics function above must depend
  // on every key that is bound to a command-line flag, which means those keys must
  // live in this package, not in the application logic above.
  // (In principle, a key not associated with a command-line flag may be declared anywhere.)

  // booleans: the +/- prefix indicates the default value (true/false).
  public static final String EXPERIMENTAL_JAVA_LIBRARY_EXPORT = "-experimental_java_library_export";
  public static final String INCOMPATIBLE_STOP_EXPORTING_LANGUAGE_MODULES =
      "-incompatible_stop_exporting_language_modules";
  public static final String INCOMPATIBLE_ALLOW_TAGS_PROPAGATION =
      "+incompatible_allow_tags_propagation";
  public static final String EXPERIMENTAL_BUILTINS_DUMMY = "-experimental_builtins_dummy";
  public static final String EXPERIMENTAL_BZL_VISIBILITY = "+experimental_bzl_visibility";
  public static final String CHECK_BZL_VISIBILITY = "+check_bzl_visibility";
  public static final String EXPERIMENTAL_CC_SHARED_LIBRARY = "-experimental_cc_shared_library";
  public static final String EXPERIMENTAL_DISABLE_EXTERNAL_PACKAGE =
      "-experimental_disable_external_package";
  public static final String EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS =
      "-experimental_enable_android_migration_apis";
  public static final String EXPERIMENTAL_ENABLE_SCL_DIALECT = "-experimental_enable_scl_dialect";
  public static final String ENABLE_BZLMOD = "-enable_bzlmod";
  public static final String EXPERIMENTAL_ISOLATED_EXTENSION_USAGES =
      "-experimental_isolated_extension_usages";
  public static final String EXPERIMENTAL_JAVA_PROTO_LIBRARY_DEFAULT_HAS_SERVICES =
      "+experimental_java_proto_library_default_has_services";
  public static final String INCOMPATIBLE_EXISTING_RULES_IMMUTABLE_VIEW =
      "+incompatible_existing_rules_immutable_view";
  public static final String EXPERIMENTAL_GOOGLE_LEGACY_API = "-experimental_google_legacy_api";
  public static final String EXPERIMENTAL_PLATFORMS_API = "-experimental_platforms_api";
  public static final String EXPERIMENTAL_REPO_REMOTE_EXEC = "-experimental_repo_remote_exec";
  public static final String EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT =
      "-experimental_sibling_repository_layout";
  public static final String EXPERIMENTAL_ACTION_RESOURCE_SET = "+experimental_action_resource_set";
  public static final String EXPERIMENTAL_LAZY_TEMPLATE_EXPANSION =
      "+experimental_lazy_template_expansion";
  public static final String EXPERIMENTAL_ANALYSIS_TEST_CALL = "+experimental_analysis_test_call";
  public static final String INCOMPATIBLE_ALWAYS_CHECK_DEPSET_ELEMENTS =
      "+incompatible_always_check_depset_elements";
  public static final String INCOMPATIBLE_DEPSET_FOR_LIBRARIES_TO_LINK_GETTER =
      "+incompatible_depset_for_libraries_to_link_getter";
  public static final String INCOMPATIBLE_DEPSET_FOR_JAVA_OUTPUT_SOURCE_JARS =
      "+incompatible_depset_for_java_output_source_jars";
  public static final String INCOMPATIBLE_DISABLE_TARGET_PROVIDER_FIELDS =
      "-incompatible_disable_target_provider_fields";
  public static final String INCOMPATIBLE_DISALLOW_EMPTY_GLOB = "-incompatible_disallow_empty_glob";
  public static final String INCOMPATIBLE_DISALLOW_STRUCT_PROVIDER_SYNTAX =
      "-incompatible_disallow_struct_provider_syntax";
  public static final String INCOMPATIBLE_PACKAGE_GROUP_HAS_PUBLIC_SYNTAX =
      FlagConstants.INCOMPATIBLE_PACKAGE_GROUP_HAS_PUBLIC_SYNTAX;
  public static final String INCOMPATIBLE_FIX_PACKAGE_GROUP_REPOROOT_SYNTAX =
      FlagConstants.INCOMPATIBLE_FIX_PACKAGE_GROUP_REPOROOT_SYNTAX;
  public static final String INCOMPATIBLE_DO_NOT_SPLIT_LINKING_CMDLINE =
      "+incompatible_do_not_split_linking_cmdline";
  public static final String INCOMPATIBLE_JAVA_COMMON_PARAMETERS =
      "+incompatible_java_common_parameters";
  public static final String INCOMPATIBLE_NEW_ACTIONS_API = "+incompatible_new_actions_api";
  public static final String INCOMPATIBLE_NO_ATTR_LICENSE = "+incompatible_no_attr_license";
  public static final String INCOMPATIBLE_NO_IMPLICIT_FILE_EXPORT =
      "-incompatible_no_implicit_file_export";
  public static final String INCOMPATIBLE_NO_RULE_OUTPUTS_PARAM =
      "-incompatible_no_rule_outputs_param";
  public static final String INCOMPATIBLE_REQUIRE_LINKER_INPUT_CC_API =
      "+incompatible_require_linker_input_cc_api";
  public static final String INCOMPATIBLE_RUN_SHELL_COMMAND_STRING =
      "+incompatible_run_shell_command_string";
  public static final String INCOMPATIBLE_STRUCT_HAS_NO_METHODS =
      "-incompatible_struct_has_no_methods";
  public static final String INCOMPATIBLE_USE_CC_CONFIGURE_FROM_RULES_CC =
      "-incompatible_use_cc_configure_from_rules";
  public static final String INCOMPATIBLE_UNAMBIGUOUS_LABEL_STRINGIFICATION =
      "+incompatible_unambiguous_label_stringification";
  public static final String INCOMPATIBLE_VISIBILITY_PRIVATE_ATTRIBUTES_AT_DEFINITION =
      "-incompatible_visibility_private_attributes_at_definition";
  public static final String INCOMPATIBLE_TOP_LEVEL_ASPECTS_REQUIRE_PROVIDERS =
      "-incompatible_top_level_aspects_require_providers";
  public static final String INCOMPATIBLE_DISABLE_STARLARK_HOST_TRANSITIONS =
      "-incompatible_disable_starlark_host_transitions";
  public static final String EXPERIMENTAL_GET_FIXED_CONFIGURED_ACTION_ENV =
      "-experimental_get_fixed_configured_action_env";

  // non-booleans
  public static final StarlarkSemantics.Key<String> EXPERIMENTAL_BUILTINS_BZL_PATH =
      new StarlarkSemantics.Key<>("experimental_builtins_bzl_path", "%bundled%");
  public static final StarlarkSemantics.Key<List<String>> EXPERIMENTAL_BUILTINS_INJECTION_OVERRIDE =
      new StarlarkSemantics.Key<>("experimental_builtins_injection_override", ImmutableList.of());
  public static final StarlarkSemantics.Key<Long> MAX_COMPUTATION_STEPS =
      new StarlarkSemantics.Key<>("max_computation_steps", 0L);
  public static final StarlarkSemantics.Key<Integer> NESTED_SET_DEPTH_LIMIT =
      new StarlarkSemantics.Key<>("nested_set_depth_limit", 3500);
}
