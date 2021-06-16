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
import java.io.Serializable;
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
 *
 * For both readability and correctness, the relative order of the options in all of these locations
 * must be kept consistent; to make it easy we use alphabetic order. The parts that need updating
 * are marked with the comment "<== Add new options here in alphabetic order ==>".
 */
public class BuildLanguageOptions extends OptionsBase implements Serializable {

  // <== Add new options here in alphabetic order ==>

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
      name = "incompatible_enable_exports_provider",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS, OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "This flag enables exports provider and JavaInfo.transitive_exports call.")
  public boolean incompatibleEnableExportsProvider;

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
      name = "experimental_ninja_actions",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help = "If set to true, enables Ninja execution functionality.")
  public boolean experimentalNinjaActions;

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
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
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
      name = "experimental_exec_groups",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.EXECUTION},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "If set to true, allows rule authors define and access multiple execution groups "
              + "during rule definition. This work is ongoing.")
  public boolean experimentalExecGroups;

  @Option(
      name = "experimental_allow_tags_propagation",
      oldName = "incompatible_allow_tags_propagation",
      defaultValue = "false",
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
      name = "incompatible_struct_has_no_methods",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
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
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
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
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "If set to true, disable the ability to access providers on 'target' objects via field "
              + "syntax. Use provider-key syntax instead. For example, instead of using "
              + "`ctx.attr.dep.my_info` to access `my_info` from inside a rule implementation "
              + "function, use `ctx.attr.dep[MyInfo]`. See "
              + "https://github.com/bazelbuild/bazel/issues/9014 for details.")
  public boolean incompatibleDisableTargetProviderFields;

  @Option(
      name = "incompatible_disable_depset_items",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "If set to true, disable the 'items' parameter of the depset constructor. Use "
              + "the 'transitive' and 'direct' parameters instead.")
  public boolean incompatibleDisableDepsetItems;

  // For Bazel, this flag is a no-op. Bazel doesn't support built-in third party license checking
  // (see https://github.com/bazelbuild/bazel/issues/7444).
  //
  // For Blaze in Google, this flag is still needed to deprecate the logic that's already been
  // removed from Bazel. That logic was introduced before Bazel existed, so Google's dependency on
  // it is deeper. But we don't want that to add unnecessary baggage to Bazel or slow down Bazel's
  // development. So this flag lets Blaze migrate on a slower timeline without blocking Bazel. This
  // means you as a Bazel user are getting better code than Google has! (for a while, at least)
  @Option(
      name = "incompatible_disable_third_party_license_checking",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = OptionEffectTag.BUILD_FILE_SEMANTICS,
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "If true, disables all license checking logic")
  public boolean incompatibleDisableThirdPartyLicenseChecking;

  @Option(
      name = "incompatible_disallow_empty_glob",
      defaultValue = "false",
      category = "incompatible changes",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "If set to true, the default value of the `allow_empty` argument of glob() is False.")
  public boolean incompatibleDisallowEmptyGlob;

  @Option(
      name = "incompatible_disallow_legacy_javainfo",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "Deprecated. No-op.")
  // TODO(elenairina): Move option to graveyard after the flag is removed from the global blazerc.
  public boolean incompatibleDisallowLegacyJavaInfo;

  @Option(
      name = "incompatible_disallow_struct_provider_syntax",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "If set to true, rule implementation functions may not return a struct. They must "
              + "instead return a list of provider instances.")
  public boolean incompatibleDisallowStructProviderSyntax;

  @Option(
      name = "incompatible_visibility_private_attributes_at_definition",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "If set to true, the visibility of private rule attributes is checked with respect "
              + "to the rule definition, rather than the rule usage.")
  public boolean incompatibleVisibilityPrivateAttributesAtDefinition;

  @Option(
      name = "incompatible_new_actions_api",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "If set to true, the API to create actions is only available on `ctx.actions`, "
              + "not on `ctx`.")
  public boolean incompatibleNewActionsApi;

  @Option(
      name = "incompatible_no_attr_license",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "If set to true, disables the function `attr.license`.")
  public boolean incompatibleNoAttrLicense;

  @Option(
      name = "incompatible_applicable_licenses",
      defaultValue = "false",
      // TODO(aiuto): change to OptionDocumentationCategory.STARLARK_SEMANTICS,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "If set to true, enables the function `attr.applicable_licenses`.")
  public boolean incompatibleApplicableLicenses;

  @Option(
      name = "incompatible_no_implicit_file_export",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
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
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "If set to true, disables the `outputs` parameter of the `rule()` Starlark function.")
  public boolean incompatibleNoRuleOutputsParam;

  @Option(
      name = "incompatible_run_shell_command_string",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
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
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
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
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "When true, Bazel will no longer allow using cc_configure from @bazel_tools. "
              + "Please see https://github.com/bazelbuild/bazel/issues/10134 for details and "
              + "migration instructions.")
  public boolean incompatibleUseCcConfigureFromRulesCc;

  @Option(
      name = "incompatible_depset_for_libraries_to_link_getter",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "When true, Bazel no longer returns a list from linking_context.libraries_to_link but "
              + "returns a depset instead.")
  public boolean incompatibleDepsetForLibrariesToLinkGetter;

  @Option(
      name = "incompatible_restrict_string_escapes",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "If set to true, unknown string escapes like `\\a` become rejected.")
  public boolean incompatibleRestrictStringEscapes;

  @Option(
      name = "incompatible_linkopts_to_linklibs",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.ACTION_COMMAND_LINES},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "If set to true the default linkopts in the default toolchain are passed as linklibs "
              + "instead of linkopts to cc_toolchain_config")
  public boolean incompatibleLinkoptsToLinklibs;

  @Option(
      name = "incompatible_java_common_parameters",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "If set to true, the jar_file, and host_javabase parameters in pack_sources and "
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
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      help =
          "If set to true, the top level aspect will honor its required providers and only run on"
              + " top level targets whose rules' advertised providers satisfy the required"
              + " providers of the aspect.")
  public boolean incompatibleTopLevelAspectsRequireProviders;

  @Option(
      name = "experimental_required_aspects",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "If set to true, allows created aspect to require a list of aspects to be propagated"
              + " before it.")
  public boolean experimentalRequiredAspects;

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
            // <== Add new options here in alphabetic order ==>
            .setBool(EXPERIMENTAL_ALLOW_TAGS_PROPAGATION, experimentalAllowTagsPropagation)
            .set(EXPERIMENTAL_BUILTINS_BZL_PATH, experimentalBuiltinsBzlPath)
            .setBool(EXPERIMENTAL_BUILTINS_DUMMY, experimentalBuiltinsDummy)
            .set(EXPERIMENTAL_BUILTINS_INJECTION_OVERRIDE, experimentalBuiltinsInjectionOverride)
            .setBool(
                EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS, experimentalEnableAndroidMigrationApis)
            .setBool(INCOMPATIBLE_ENABLE_EXPORTS_PROVIDER, incompatibleEnableExportsProvider)
            .setBool(EXPERIMENTAL_GOOGLE_LEGACY_API, experimentalGoogleLegacyApi)
            .setBool(EXPERIMENTAL_NINJA_ACTIONS, experimentalNinjaActions)
            .setBool(EXPERIMENTAL_PLATFORMS_API, experimentalPlatformsApi)
            .setBool(EXPERIMENTAL_CC_SHARED_LIBRARY, experimentalCcSharedLibrary)
            .setBool(EXPERIMENTAL_REPO_REMOTE_EXEC, experimentalRepoRemoteExec)
            .setBool(EXPERIMENTAL_DISABLE_EXTERNAL_PACKAGE, experimentalDisableExternalPackage)
            .setBool(EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT, experimentalSiblingRepositoryLayout)
            .setBool(EXPERIMENTAL_EXEC_GROUPS, experimentalExecGroups)
            .setBool(INCOMPATIBLE_APPLICABLE_LICENSES, incompatibleApplicableLicenses)
            .setBool(
                INCOMPATIBLE_DISABLE_TARGET_PROVIDER_FIELDS,
                incompatibleDisableTargetProviderFields)
            .setBool(
                INCOMPATIBLE_DISABLE_THIRD_PARTY_LICENSE_CHECKING,
                incompatibleDisableThirdPartyLicenseChecking)
            .setBool(
                INCOMPATIBLE_ALWAYS_CHECK_DEPSET_ELEMENTS, incompatibleAlwaysCheckDepsetElements)
            .setBool(INCOMPATIBLE_DISABLE_DEPSET_ITEMS, incompatibleDisableDepsetItems)
            .setBool(INCOMPATIBLE_DISALLOW_EMPTY_GLOB, incompatibleDisallowEmptyGlob)
            .setBool(
                INCOMPATIBLE_DISALLOW_STRUCT_PROVIDER_SYNTAX,
                incompatibleDisallowStructProviderSyntax)
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
                INCOMPATIBLE_DEPSET_FOR_LIBRARIES_TO_LINK_GETTER,
                incompatibleDepsetForLibrariesToLinkGetter)
            .setBool(INCOMPATIBLE_REQUIRE_LINKER_INPUT_CC_API, incompatibleRequireLinkerInputCcApi)
            .setBool(INCOMPATIBLE_RESTRICT_STRING_ESCAPES, incompatibleRestrictStringEscapes)
            .setBool(INCOMPATIBLE_LINKOPTS_TO_LINKLIBS, incompatibleLinkoptsToLinklibs)
            .set(MAX_COMPUTATION_STEPS, maxComputationSteps)
            .set(NESTED_SET_DEPTH_LIMIT, nestedSetDepthLimit)
            .setBool(
                INCOMPATIBLE_TOP_LEVEL_ASPECTS_REQUIRE_PROVIDERS,
                incompatibleTopLevelAspectsRequireProviders)
            .setBool(EXPERIMENTAL_REQUIRED_ASPECTS, experimentalRequiredAspects)
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
  public static final String EXPERIMENTAL_ALLOW_TAGS_PROPAGATION =
      "-experimental_allow_tags_propagation";
  public static final String EXPERIMENTAL_BUILTINS_DUMMY = "-experimental_builtins_dummy";
  public static final String EXPERIMENTAL_CC_SHARED_LIBRARY = "-experimental_cc_shared_library";
  public static final String EXPERIMENTAL_DISABLE_EXTERNAL_PACKAGE =
      "-experimental_disable_external_package";
  public static final String EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS =
      "-experimental_enable_android_migration_apis";
  public static final String INCOMPATIBLE_ENABLE_EXPORTS_PROVIDER =
      "-incompatible_enable_exports_provider";
  public static final String EXPERIMENTAL_EXEC_GROUPS = "+experimental_exec_groups";
  public static final String EXPERIMENTAL_GOOGLE_LEGACY_API = "-experimental_google_legacy_api";
  public static final String EXPERIMENTAL_NINJA_ACTIONS = "-experimental_ninja_actions";
  public static final String EXPERIMENTAL_PLATFORMS_API = "-experimental_platforms_api";
  public static final String EXPERIMENTAL_REPO_REMOTE_EXEC = "-experimental_repo_remote_exec";
  public static final String EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT =
      "-experimental_sibling_repository_layout";
  public static final String INCOMPATIBLE_ALLOW_TAGS_PROPAGATION =
      "-incompatible_allow_tags_propagation";
  public static final String INCOMPATIBLE_ALWAYS_CHECK_DEPSET_ELEMENTS =
      "+incompatible_always_check_depset_elements";
  public static final String INCOMPATIBLE_APPLICABLE_LICENSES = "-incompatible_applicable_licenses";
  public static final String INCOMPATIBLE_DEPSET_FOR_LIBRARIES_TO_LINK_GETTER =
      "+incompatible_depset_for_libraries_to_link_getter";
  public static final String INCOMPATIBLE_DISABLE_DEPSET_ITEMS =
      "-incompatible_disable_depset_items";
  public static final String INCOMPATIBLE_DISABLE_TARGET_PROVIDER_FIELDS =
      "-incompatible_disable_target_provider_fields";
  public static final String INCOMPATIBLE_DISABLE_THIRD_PARTY_LICENSE_CHECKING =
      "+incompatible_disable_third_party_license_checking";
  public static final String INCOMPATIBLE_DISALLOW_EMPTY_GLOB = "-incompatible_disallow_empty_glob";
  public static final String INCOMPATIBLE_DISALLOW_STRUCT_PROVIDER_SYNTAX =
      "-incompatible_disallow_struct_provider_syntax";
  public static final String INCOMPATIBLE_DO_NOT_SPLIT_LINKING_CMDLINE =
      "+incompatible_do_not_split_linking_cmdline";
  public static final String INCOMPATIBLE_JAVA_COMMON_PARAMETERS =
      "-incompatible_java_common_parameters";
  public static final String INCOMPATIBLE_LINKOPTS_TO_LINKLIBS =
      "+incompatible_linkopts_to_linklibs";
  public static final String INCOMPATIBLE_NEW_ACTIONS_API = "+incompatible_new_actions_api";
  public static final String INCOMPATIBLE_NO_ATTR_LICENSE = "+incompatible_no_attr_license";
  public static final String INCOMPATIBLE_NO_IMPLICIT_FILE_EXPORT =
      "-incompatible_no_implicit_file_export";
  public static final String INCOMPATIBLE_NO_RULE_OUTPUTS_PARAM =
      "-incompatible_no_rule_outputs_param";
  public static final String INCOMPATIBLE_REQUIRE_LINKER_INPUT_CC_API =
      "+incompatible_require_linker_input_cc_api";
  public static final String INCOMPATIBLE_RESTRICT_STRING_ESCAPES =
      "+incompatible_restrict_string_escapes";
  public static final String INCOMPATIBLE_RUN_SHELL_COMMAND_STRING =
      "+incompatible_run_shell_command_string";
  public static final String INCOMPATIBLE_STRUCT_HAS_NO_METHODS =
      "-incompatible_struct_has_no_methods";
  public static final String INCOMPATIBLE_USE_CC_CONFIGURE_FROM_RULES_CC =
      "-incompatible_use_cc_configure_from_rules";
  public static final String INCOMPATIBLE_VISIBILITY_PRIVATE_ATTRIBUTES_AT_DEFINITION =
      "-incompatible_visibility_private_attributes_at_definition";
  public static final String INCOMPATIBLE_TOP_LEVEL_ASPECTS_REQUIRE_PROVIDERS =
      "-incompatible_top_level_aspects_require_providers";
  public static final String EXPERIMENTAL_REQUIRED_ASPECTS = "-experimental_required_aspects";

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
