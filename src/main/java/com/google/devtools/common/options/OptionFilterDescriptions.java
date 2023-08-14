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
package com.google.devtools.common.options;

import com.google.common.collect.ImmutableMap;

/**
 * Provides descriptions of the options filters, for use in generated documentation and usage text.
 */
public class OptionFilterDescriptions {

  /** The order that the categories should be listed in. */
  static OptionDocumentationCategory[] documentationOrder = {
    OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
    OptionDocumentationCategory.EXECUTION_STRATEGY,
    OptionDocumentationCategory.TOOLCHAIN,
    OptionDocumentationCategory.OUTPUT_SELECTION,
    OptionDocumentationCategory.OUTPUT_PARAMETERS,
    OptionDocumentationCategory.INPUT_STRICTNESS,
    OptionDocumentationCategory.SIGNING,
    OptionDocumentationCategory.STARLARK_SEMANTICS,
    OptionDocumentationCategory.TESTING,
    OptionDocumentationCategory.QUERY,
    OptionDocumentationCategory.MOD_COMMAND,
    OptionDocumentationCategory.BZLMOD,
    OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
    OptionDocumentationCategory.LOGGING,
    OptionDocumentationCategory.GENERIC_INPUTS,
    OptionDocumentationCategory.REMOTE,
    OptionDocumentationCategory.UNCATEGORIZED
  };

  static ImmutableMap<OptionDocumentationCategory, String> getOptionCategoriesEnumDescription(
      String productName) {
    ImmutableMap.Builder<OptionDocumentationCategory, String> optionCategoriesBuilder =
        ImmutableMap.builder();
    optionCategoriesBuilder
        .put(
            OptionDocumentationCategory.UNCATEGORIZED,
            "Miscellaneous options, not otherwise categorized.")
        .put( // Here for completeness, the help output should not include this option.
            OptionDocumentationCategory.UNDOCUMENTED,
            "This feature should not be documented, as it is not meant for general use")
        .put(
            OptionDocumentationCategory.BAZEL_CLIENT_OPTIONS,
            "Options that appear before the command and are parsed by the client")
        .put(
            OptionDocumentationCategory.LOGGING,
            "Options that affect the verbosity, format or location of logging")
        .put(OptionDocumentationCategory.EXECUTION_STRATEGY, "Options that control build execution")
        .put(
            OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
            "Options that trigger optimizations of the build time")
        .put(
            OptionDocumentationCategory.OUTPUT_SELECTION,
            "Options that control the output of the command")
        .put(
            OptionDocumentationCategory.OUTPUT_PARAMETERS,
            "Options that let the user configure the intended output, affecting its value, as "
                + "opposed to its existence")
        .put(
            OptionDocumentationCategory.INPUT_STRICTNESS,
            "Options that affect how strictly Bazel enforces valid build inputs (rule definitions, "
                + " flag combinations, etc.)")
        .put(
            OptionDocumentationCategory.SIGNING,
            "Options that affect the signing outputs of a build")
        .put(
            OptionDocumentationCategory.STARLARK_SEMANTICS,
            "This option affects semantics of the Starlark language or the build API accessible to "
                + "BUILD files, .bzl files, or WORKSPACE files.")
        .put(
            OptionDocumentationCategory.TESTING,
            "Options that govern the behavior of the test environment or test runner")
        .put(
            OptionDocumentationCategory.TOOLCHAIN,
            "Options that configure the toolchain used for action execution")
        .put(OptionDocumentationCategory.QUERY, "Options relating to query output and semantics")
        .put(
            OptionDocumentationCategory.MOD_COMMAND,
            "Options relating to the output and semantics of the `mod` subcommand")
        .put(OptionDocumentationCategory.BZLMOD, "Options relating to Bzlmod output and semantics")
        .put(
            OptionDocumentationCategory.GENERIC_INPUTS,
            "Options specifying or altering a generic input to a Bazel command that does not fall "
                + "into other categories.")
        .put(OptionDocumentationCategory.REMOTE, "Remote caching and execution options");
    return optionCategoriesBuilder.build();
  }

  public static ImmutableMap<OptionEffectTag, String> getOptionEffectTagDescription(
      String productName) {
    ImmutableMap.Builder<OptionEffectTag, String> effectTagDescriptionBuilder =
        ImmutableMap.builder();
    effectTagDescriptionBuilder
        .put(OptionEffectTag.UNKNOWN, "This option has unknown, or undocumented, effect.")
        .put(OptionEffectTag.NO_OP, "This option has literally no effect.")
        .put(
            OptionEffectTag.LOSES_INCREMENTAL_STATE,
            "Changing the value of this option can cause significant loss of incremental "
                + "state, which slows builds. State could be lost due to a server restart or to "
                + "invalidation of a large part of the dependency graph.")
        .put(
            OptionEffectTag.CHANGES_INPUTS,
            "This option actively changes the inputs that "
                + productName
                + " considers for the build, such as filesystem restrictions, repository versions, "
                + "or other options.")
        .put(
            OptionEffectTag.AFFECTS_OUTPUTS,
            "This option affects "
                + productName
                + "'s outputs. This tag is intentionally broad, can include transitive affects, "
                + "and does not specify the type of output it affects.")
        .put(
            OptionEffectTag.BUILD_FILE_SEMANTICS,
            "This option affects the semantics of BUILD or .bzl files.")
        .put(
            OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION,
            "This option affects settings of "
                + productName
                + "-internal machinery. This tag does not, on its own, mean that build artifacts "
                + "are affected.")
        .put(
            OptionEffectTag.LOADING_AND_ANALYSIS,
            "This option affects the loading and analysis of dependencies, and the building "
                + "of the dependency graph.")
        .put(
            OptionEffectTag.EXECUTION,
            "This option affects the execution phase, such as sandboxing or remote execution "
                + "related options.")
        .put(
            OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS,
            "This option triggers an optimization that may be machine specific and is not "
                + "guaranteed to work on all machines. The optimization could include a tradeoff "
                + "with other aspects of performance, such as memory or cpu cost.")
        .put(
            OptionEffectTag.EAGERNESS_TO_EXIT,
            "This option changes how eagerly "
                + productName
                + " will exit from a failure, where a choice between continuing despite the "
                + "failure and ending the invocation exists.")
        .put(
            OptionEffectTag.BAZEL_MONITORING,
            "This option is used to monitor " + productName + "'s behavior and performance.")
        .put(
            OptionEffectTag.TERMINAL_OUTPUT,
            "This option affects " + productName + "'s terminal output.")
        .put(
            OptionEffectTag.ACTION_COMMAND_LINES,
            "This option changes the command line arguments of one or more build actions.")
        .put(
            OptionEffectTag.TEST_RUNNER,
            "This option changes the testrunner environment of the build.");
    return effectTagDescriptionBuilder.build();
  }

  public static ImmutableMap<OptionMetadataTag, String> getOptionMetadataTagDescription(
      String productName) {
    ImmutableMap.Builder<OptionMetadataTag, String> effectTagDescriptionBuilder =
        ImmutableMap.builder();
    effectTagDescriptionBuilder
        .put(
            OptionMetadataTag.EXPERIMENTAL,
            "This option triggers an experimental feature with no guarantees of functionality.")
        .put(
            OptionMetadataTag.INCOMPATIBLE_CHANGE,
            "This option triggers a breaking change. Use this option to test your migration "
                + "readiness or get early access to the new feature")
        .put(
            OptionMetadataTag.DEPRECATED,
            "This option is deprecated. It might be that the feature it affects is deprecated, "
                + "or that another method of supplying the information is preferred.")
        .put(
            OptionMetadataTag.HIDDEN, // Here for completeness, these options are UNDOCUMENTED.
            "This option should not be used by a user, and should not be logged.")
        .put(
            OptionMetadataTag.INTERNAL, // Here for completeness, these options are UNDOCUMENTED.
            "This option isn't even a option, and should not be logged.")
        .put(
            OptionMetadataTag.EXPLICIT_IN_OUTPUT_PATH,
            "This option is explicitly mentioned in the output directory.");
    return effectTagDescriptionBuilder.build();
  }
}
