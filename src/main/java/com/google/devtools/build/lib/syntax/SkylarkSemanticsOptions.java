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

package com.google.devtools.build.lib.syntax;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.UsesOnlyCoreTypes;
import java.io.Serializable;

/**
 * Contains options that affect Skylark's semantics.
 *
 * <p>These are injected into Skyframe when a new build invocation occurs. Changing these options
 * between builds will trigger a reevaluation of everything that depends on the Skylark
 * interpreter &mdash; in particular, processing BUILD and .bzl files.
 *
 * <p>Because these options are stored in Skyframe, they must be immutable and serializable, and so
 * are subject to the restrictions of {@link UsesOnlyCoreTypes}: No {@link Option#allowMultiple}
 * options, and no options with types not handled by the default converters. (Technically all
 * options classes are mutable because their fields are public and non-final, but we assume no one
 * is manipulating these fields by the time parsing is complete.)
 */
@UsesOnlyCoreTypes
public class SkylarkSemanticsOptions extends OptionsBase implements Serializable {

  /** Used in an integration test to confirm that flags are visible to the interpreter. */
  @Option(
    name = "internal_skylark_flag_test_canary",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.UNKNOWN}
  )
  public boolean internalSkylarkFlagTestCanary;

  /**
   * Used in testing to produce a truly minimalistic Extension object for certain evaluation
   * contexts. This flag is Bazel-specific.
   */
  // TODO(bazel-team): A pending incompatible change will make it so that load()ed and built-in
  // symbols do not get re-exported, making this flag obsolete.
  @Option(
    name = "internal_do_not_export_builtins",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.UNKNOWN}
  )
  public boolean internalDoNotExportBuiltins;

  @Option(
    name = "incompatible_disallow_set_constructor",
    defaultValue = "false",
    category = "incompatible changes",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
    help = "If set to true, disables the deprecated `set` constructor for depsets."
  )
  public boolean incompatibleDisallowSetConstructor;

  @Option(
    name = "incompatible_disallow_keyword_only_args",
    defaultValue = "false",
    category = "incompatible changes",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
    help = "If set to true, disables the keyword-only argument syntax in function definition."
  )
  public boolean incompatibleDisallowKeywordOnlyArgs;

  @Option(
    name = "incompatible_list_plus_equals_inplace",
    defaultValue = "false",
    category = "incompatible changes",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
    help =
        "If set to true, `+=` on lists works like the `extend` method mutating the original "
            + "list. Otherwise it copies the original list without mutating it."
  )
  public boolean incompatibleListPlusEqualsInplace;

  @Option(
    name = "incompatible_disallow_dict_plus",
    defaultValue = "false",
    category = "incompatible changes",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
    help = "If set to true, the `+` becomes disabled for dicts."
  )
  public boolean incompatibleDisallowDictPlus;

  @Option(
    name = "incompatible_bzl_disallow_load_after_statement",
    defaultValue = "false",
    category = "incompatible changes",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
    help =
        "If set to true, all `load` must be called at the top of .bzl files, before any other "
            + "statement."
  )
  public boolean incompatibleBzlDisallowLoadAfterStatement;

  @Option(
    name = "incompatible_load_argument_is_label",
    defaultValue = "false",
    category = "incompatible changes",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
    help =
        "If set to true, the first argument of 'load' statements is a label (not a path). "
            + "It must start with '//' or ':'."
  )
  public boolean incompatibleLoadArgumentIsLabel;

  @Option(
    name = "incompatible_disallow_toplevel_if_statement",
    defaultValue = "true",
    category = "incompatible changes",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
    help =
        "If set to true, 'if' statements are forbidden at the top-level "
            + "(outside a function definition)"
  )
  public boolean incompatibleDisallowToplevelIfStatement;

  @Option(
    name = "incompatible_comprehension_variables_do_not_leak",
    defaultValue = "false",
    category = "incompatible changes",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
    help =
        "If set to true, loop variables in a comprehension shadow any existing variable by "
            + "the same name. If the existing variable was declared in the same scope that "
            + "contains the comprehension, then it also becomes inaccessible after the "
            + " comprehension executes."
  )
  public boolean incompatibleComprehensionVariablesDoNotLeak;

  @Option(
    name = "incompatible_depset_is_not_iterable",
    defaultValue = "false",
    category = "incompatible changes",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
    help =
        "If set to true, depset type is not iterable. For loops and functions expecting an "
            + "iterable will reject depset objects. Use the `.to_list` method to explicitly "
            + "convert to a list."
  )
  public boolean incompatibleDepsetIsNotIterable;

  @Option(
    name = "incompatible_string_is_not_iterable",
    defaultValue = "false",
    category = "incompatible changes",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
    help =
        "If set to true, iterating over a string will throw an error. String indexing and `len` "
            + "are still allowed."
  )
  public boolean incompatibleStringIsNotIterable;

  @Option(
    name = "incompatible_dict_literal_has_no_duplicates",
    defaultValue = "false",
    category = "incompatible changes",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
    help = "If set to true, the dictionary literal syntax doesn't allow duplicated keys."
  )
  public boolean incompatibleDictLiteralHasNoDuplicates;

  @Option(
      name = "incompatible_new_actions_api",
      defaultValue = "false",
      category = "incompatible changes",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help = "If set to true, the API to create actions is only available on `ctx.actions`, "
          + "not on `ctx`."
  )
  public boolean incompatibleNewActionsApi;

  @Option(
    name = "incompatible_checked_arithmetic",
    defaultValue = "false",
    category = "incompatible changes",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
    help = "If set to true, arithmetic operations throw an error in case of overflow/underflow."
  )
  public boolean incompatibleCheckedArithmetic;

  @Option(
    name = "incompatible_descriptive_string_representations",
    defaultValue = "false",
    category = "incompatible changes",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
    help =
        "If set to true, objects are converted to strings by `str` and `repr` functions using the "
            + "new style representations that are designed to be more descriptive and not to leak "
            + "information that's not supposed to be exposed."
  )
  public boolean incompatibleDescriptiveStringRepresentations;
}
