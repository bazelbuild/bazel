// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.syntax.SkylarkSemantics;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;
import java.io.Serializable;
import java.util.List;

/**
 * Contains options that affect Skylark's semantics.
 *
 * <p>These are injected into Skyframe (as an instance of {@link SkylarkSemantics}) when a new build
 * invocation occurs. Changing these options between builds will therefore trigger a reevaluation of
 * everything that depends on the Skylark interpreter &mdash; in particular, evaluation of all BUILD
 * and .bzl files.
 *
 * <p><em>To add a new option, update the following:</em>
 * <ul>
 *   <li>Add a new abstract method (which is interpreted by {@code AutoValue} as a field) to {@link
 *       SkylarkSemantics} and {@link SkylarkSemantics.Builder}. Set its default value in {@link
 *       SkylarkSemantics#DEFAULT_SEMANTICS}.
 *
 *   <li>Add a new {@code @Option}-annotated field to this class. The field name and default value
 *       should be the same as in {@link SkylarkSemantics}, and the option name in the annotation
 *       should be that name written in snake_case. Add a line to set the new field in {@link
 *       #toSkylarkSemantics}.
 *
 *   <li>Add a line to set the new field in both {@link
 *       SkylarkSemanticsConsistencyTest#buildRandomOptions} and {@link
 *       SkylarkSemanticsConsistencyTest#buildRandomSemantics}.
 *
 *   <li>Update manual documentation in site/docs/skylark/backward-compatibility.md. Also remember
 *       to update this when flipping a flag's default value.
 *
 *   <li>Boolean semantic flags can toggle Skylark methods on or off. To do this, add a new entry
 *       to {@link SkylarkSemantics#FlagIdentifier}. Then, specify the identifier in
 *       {@code SkylarkCallable.enableOnlyWithFlag} or {@code SkylarkCallable.disableWithFlag}.
 * </ul>
 * For both readability and correctness, the relative order of the options in all of these locations
 * must be kept consistent; to make it easy we use alphabetic order. The parts that need updating
 * are marked with the comment "<== Add new options here in alphabetic order ==>".
 */
public class SkylarkSemanticsOptions extends OptionsBase implements Serializable {

  // <== Add new options here in alphabetic order ==>

  @Option(
      name = "experimental_analysis_testing_improvements",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {
          OptionEffectTag.BUILD_FILE_SEMANTICS,
          OptionEffectTag.LOADING_AND_ANALYSIS
      },
      metadataTags = {
          OptionMetadataTag.EXPERIMENTAL
      },
      help = "If true, enables pieces of experimental Skylark API for analysis-phase testing."
  )
  public boolean experimentalAnalysisTestingImprovements;

  @Option(
      name = "experimental_cc_skylark_api_enabled_packages",
      converter = CommaSeparatedOptionListConverter.class,
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "Passes list of packages that can use the C++ Skylark API. Don't enable this flag yet, "
              + "we will be making breaking changes.")
  public List<String> experimentalCcSkylarkApiEnabledPackages;

  @Option(
      name = "experimental_enable_android_migration_apis",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = OptionEffectTag.BUILD_FILE_SEMANTICS,
      help = "If set to true, enables the APIs required to support the Android Starlark migration.")
  public boolean experimentalEnableAndroidMigrationApis;

  @Option(
      name = "experimental_enable_repo_mapping",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = OptionEffectTag.BUILD_FILE_SEMANTICS,
      help = "If set to true, enables the use of the `repo_mapping` attribute in WORKSPACE files.")
  public boolean experimentalEnableRepoMapping;

  @Option(
      name = "experimental_remap_main_repo",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = OptionEffectTag.LOADING_AND_ANALYSIS,
      help = "If set to true, will treat references to '@<main repo name>' the same as '@'.")
  public boolean experimentalRemapMainRepo;

  @Option(
      name = "experimental_platforms_api",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help = "If set to true, enables a number of platform-related Starlark APIs useful for "
          + "debugging.")
  public boolean experimentalPlatformsApi;

  // TODO(cparsons): Resolve and finalize the transition() API. The transition implementation
  // function should accept two mandatory parameters, 'settings' and 'attr'.
  @Option(
      name = "experimental_starlark_config_transitions",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "If set to true, enables creation of configuration transition objects (the "
              + "`transition()` function) in Starlark.")
  public boolean experimentalStarlarkConfigTransitions;

  @Option(
    name = "incompatible_bzl_disallow_load_after_statement",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
    effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
    metadataTags = {
      OptionMetadataTag.INCOMPATIBLE_CHANGE,
      OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
    },
    help =
        "If set to true, all `load` must be called at the top of .bzl files, before any other "
            + "statement."
  )
  public boolean incompatibleBzlDisallowLoadAfterStatement;

  @Option(
    name = "incompatible_depset_union",
    defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
    metadataTags = {
      OptionMetadataTag.INCOMPATIBLE_CHANGE,
      OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
    },
    help =
        "If set to true, depset union using `+`, `|` or `.union` are forbidden. "
            + "Use the `depset` constructor instead."
  )
  public boolean incompatibleDepsetUnion;

  @Option(
    name = "incompatible_depset_is_not_iterable",
    defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
    metadataTags = {
      OptionMetadataTag.INCOMPATIBLE_CHANGE,
      OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
    },
    help =
        "If set to true, depset type is not iterable. For loops and functions expecting an "
            + "iterable will reject depset objects. Use the `.to_list` method to explicitly "
            + "convert to a list."
  )
  public boolean incompatibleDepsetIsNotIterable;

  @Option(
      name = "incompatible_disable_deprecated_attr_params",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "If set to true, disable the deprecated parameters 'single_file' and 'non_empty' on "
              + "skylark attribute definition methods, such as attr.label()."
  )
  public boolean incompatibleDisableDeprecatedAttrParams;

  @Option(
    name = "incompatible_disable_objc_provider_resources",
    defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
    metadataTags = {
      OptionMetadataTag.INCOMPATIBLE_CHANGE,
      OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
    },
    help = "If set to true, disallow use of deprecated resource fields on the Objc provider."
  )
  public boolean incompatibleDisableObjcProviderResources;

  @Option(
      name = "incompatible_disallow_conflicting_providers",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "If set to true, disallow rule implementation functions from returning multiple "
          + "instances of the same type of provider. (If false, only the last in the list will be "
          + "used.)"
  )
  public boolean incompatibleDisallowConflictingProviders;

  @Option(
      name = "incompatible_disallow_data_transition",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "If set to true, rule attributes cannot set 'cfg = \"data\"', which is a noop."
  )
  public boolean incompatibleDisallowDataTransition;

  @Option(
      name = "incompatible_disallow_dict_plus",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "If set to true, the `+` becomes disabled for dicts."
  )
  public boolean incompatibleDisallowDictPlus;

  @Option(
      name = "incompatible_disallow_filetype",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "If set to true, function `FileType` is not available."
  )
  public boolean incompatibleDisallowFileType;

  @Option(
      name = "incompatible_disallow_legacy_javainfo",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "If set to true, old-style JavaInfo provider construction is disallowed."
  )
  public boolean incompatibleDisallowLegacyJavaInfo;

  @Option(
      name = "incompatible_disallow_load_labels_to_cross_package_boundaries",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "If set to true, the label argument to 'load' cannot cross a package boundary."
  )
  public boolean incompatibleDisallowLoadLabelsToCrossPackageBoundaries;

  @Option(
      name = "incompatible_generate_javacommon_source_jar",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "If set to true, java_common.compile will always generate an output source jar."
  )
  public boolean incompatibleGenerateJavaCommonSourceJar;

  @Option(
    name = "incompatible_disallow_slash_operator",
    defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
    metadataTags = {
      OptionMetadataTag.INCOMPATIBLE_CHANGE,
      OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
    },
    help = "If set to true, the `/` operator is disabled. Use `//` for integer division."
  )
  public boolean incompatibleDisallowSlashOperator;

  /** Controls legacy arguments to ctx.actions.Args#add. */
  @Option(
      name = "incompatible_disallow_old_style_args_add",
      defaultValue = "false",
      category = "incompatible changes",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "If set to true, vectorized calls to Args#add are disallowed."
  )
  public boolean incompatibleDisallowOldStyleArgsAdd;

  @Option(
      name = "incompatible_expand_directories",
      defaultValue = "false",
      category = "incompatible changes",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "Controls whether directories are expanded to the list of files under that directory "
              + "when added to Args, instead of replaced by the path of the directory.")
  public boolean incompatibleExpandDirectories;

  @Option(
      name = "incompatible_new_actions_api",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "If set to true, the API to create actions is only available on `ctx.actions`, "
              + "not on `ctx`."
  )
  public boolean incompatibleNewActionsApi;

  @Option(
      name = "incompatible_no_output_attr_default",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "If set to true, disables the `default` parameter of the `attr.output` and "
          + "`attr.output_list` attribute definition functions.")
  public boolean incompatibleNoOutputAttrDefault;

  @Option(
      name = "incompatible_no_support_tools_in_action_inputs",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "If set to true, tools should be passed to `ctx.actions.run()` and "
              + "`ctx.actions.run_shell()` using the `tools` parameter instead of the `inputs` "
              + "parameter. Furthermore, if this flag is set and a `tools` parameter is not "
              + "passed to the action, it is an error for any tools to appear in the `inputs`.")
  public boolean incompatibleNoSupportToolsInActionInputs;

  @Option(
      name = "incompatible_no_target_output_group",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "If set to true, disables the output_group field of the 'Target' Starlark type.")
  public boolean incompatibleNoTargetOutputGroup;

  @Option(
      name = "incompatible_no_transitive_loads",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "If set to true, only symbols explicitly defined in the file can be loaded; "
              + "symbols introduced by load are not implicitly re-exported.")
  public boolean incompatibleNoTransitiveLoads;

  @Option(
      name = "incompatible_package_name_is_a_function",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "If set to true, the values PACKAGE_NAME and REPOSITORY_NAME are not available. "
              + "Use the package_name() or repository_name() functions instead.")
  public boolean incompatiblePackageNameIsAFunction;

  @Option(
      name = "incompatible_range_type",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "If set to true, range() will use the 'range' type instead of 'list'."
  )
  public boolean incompatibleRangeType;

  @Option(
      name = "incompatible_remove_native_git_repository",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "If set to true, the native git_repository rules are disabled; only the skylark version "
              + "will be available"
  )
  public boolean incompatibleRemoveNativeGitRepository;

  @Option(
      name = "incompatible_remove_native_http_archive",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "If set to true, the native http_archive rules are disabled; only the skylark version "
              + "will be available"
  )
  public boolean incompatibleRemoveNativeHttpArchive;

  @Option(
      name = "incompatible_static_name_resolution",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "If set to true, the interpreter follows the semantics related to name resolution, "
              + "scoping, and shadowing, as defined in "
              + "https://github.com/bazelbuild/proposals/blob/master/docs/2018-06-18-name-resolution.md")
  public boolean incompatibleStaticNameResolution;

  @Option(
      name = "incompatible_string_is_not_iterable",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.SKYLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "If set to true, iterating over a string will throw an error. String indexing and `len` "
              + "are still allowed."
  )
  public boolean incompatibleStringIsNotIterable;

  /** Used in an integration test to confirm that flags are visible to the interpreter. */
  @Option(
    name = "internal_skylark_flag_test_canary",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.UNKNOWN}
  )
  public boolean internalSkylarkFlagTestCanary;

  @Option(
      name = "incompatible_never_use_embedded_jdk_for_javabase",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "Don't use the embedded JDK as a fall-back --javabase if we can't find a locally"
              + " installed JDK (using JAVA_HOME or `which javac`).")
  public boolean incompatibleNeverUseEmbeddedJDKForJavabase;

  /** Constructs a {@link SkylarkSemantics} object corresponding to this set of option values. */
  public SkylarkSemantics toSkylarkSemantics() {
    return SkylarkSemantics.builder()
        // <== Add new options here in alphabetic order ==>
        .experimentalAnalysisTestingImprovements(experimentalAnalysisTestingImprovements)
        .experimentalCcSkylarkApiEnabledPackages(experimentalCcSkylarkApiEnabledPackages)
        .experimentalEnableAndroidMigrationApis(experimentalEnableAndroidMigrationApis)
        .experimentalEnableRepoMapping(experimentalEnableRepoMapping)
        .experimentalRemapMainRepo(experimentalRemapMainRepo)
        .experimentalPlatformsApi(experimentalPlatformsApi)
        .experimentalStarlarkConfigTransitions(experimentalStarlarkConfigTransitions)
        .incompatibleBzlDisallowLoadAfterStatement(incompatibleBzlDisallowLoadAfterStatement)
        .incompatibleDepsetIsNotIterable(incompatibleDepsetIsNotIterable)
        .incompatibleDepsetUnion(incompatibleDepsetUnion)
        .incompatibleDisableDeprecatedAttrParams(incompatibleDisableDeprecatedAttrParams)
        .incompatibleDisableObjcProviderResources(incompatibleDisableObjcProviderResources)
        .incompatibleDisallowConflictingProviders(incompatibleDisallowConflictingProviders)
        .incompatibleDisallowDataTransition(incompatibleDisallowDataTransition)
        .incompatibleDisallowDictPlus(incompatibleDisallowDictPlus)
        .incompatibleDisallowFileType(incompatibleDisallowFileType)
        .incompatibleDisallowLegacyJavaInfo(incompatibleDisallowLegacyJavaInfo)
        .incompatibleDisallowLoadLabelsToCrossPackageBoundaries(
            incompatibleDisallowLoadLabelsToCrossPackageBoundaries)
        .incompatibleDisallowOldStyleArgsAdd(incompatibleDisallowOldStyleArgsAdd)
        .incompatibleDisallowSlashOperator(incompatibleDisallowSlashOperator)
        .incompatibleExpandDirectories(incompatibleExpandDirectories)
        .incompatibleGenerateJavaCommonSourceJar(incompatibleGenerateJavaCommonSourceJar)
        .incompatibleNewActionsApi(incompatibleNewActionsApi)
        .incompatibleNoOutputAttrDefault(incompatibleNoOutputAttrDefault)
        .incompatibleNoSupportToolsInActionInputs(incompatibleNoSupportToolsInActionInputs)
        .incompatibleNoTargetOutputGroup(incompatibleNoTargetOutputGroup)
        .incompatibleNoTransitiveLoads(incompatibleNoTransitiveLoads)
        .incompatiblePackageNameIsAFunction(incompatiblePackageNameIsAFunction)
        .incompatibleRangeType(incompatibleRangeType)
        .incompatibleRemoveNativeGitRepository(incompatibleRemoveNativeGitRepository)
        .incompatibleRemoveNativeHttpArchive(incompatibleRemoveNativeHttpArchive)
        .incompatibleStaticNameResolution(incompatibleStaticNameResolution)
        .incompatibleStringIsNotIterable(incompatibleStringIsNotIterable)
        .internalSkylarkFlagTestCanary(internalSkylarkFlagTestCanary)
        .incompatibleNeverUseEmbeddedJDKForJavabase(incompatibleNeverUseEmbeddedJDKForJavabase)
        .build();
  }
}
