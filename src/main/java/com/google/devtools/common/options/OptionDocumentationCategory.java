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

/**
 * These categories are used to logically group options in generated documentation, both the command
 * line output for the standard HelpCommand and the html output used for command-line-reference on
 * the website.
 *
 * <p>Constraints for adding new categories:
 *
 * <ul>
 *   <li>Since these are for grouping, we want useful sizes of groups. Somewhere between 5 and 200
 *       (ok, maybe less than that) options, probably. A category for 2 options is pretty useless,
 *       and a category for all options equally so.
 *   <li>Each option needs to belong to exactly one of these groups, so the categories should be
 *       clearly distinct.
 *   <li>These are easy to change, and not brittle, so feel free to add new ones. However, if you
 *       add a new category, other flags that used to be categorized in some related way but belong
 *       in the new category should be updated to keep our docs fresh. Either do it yourself or file
 *       a bug against the owners of these flags.
 * </ul>
 */
public enum OptionDocumentationCategory {
  /**
   * A category to aid transition, to make it obvious that an option needs to be categorized. Note:
   * Do NOT add this to new flags!
   */
  UNCATEGORIZED,

  /**
   * A category for flags that are intended to not be listed, and for whom a documentation category
   * does not make sense.
   */
  UNDOCUMENTED,

  /**
   * Startup options appear before the command and are parsed by the client. Changing them may cause
   * a server restart, see OptionEffectTag.LOSES_INCREMENTAL_STATE.
   */
  BAZEL_CLIENT_OPTIONS,

  /** This option's primary purpose is to affect the verbosity, format or location of logging. */
  LOGGING,

  /**
   * This option affects how strictly Bazel enforces valid build inputs (rule definitions, flag
   * combinations, etc).
   */
  INPUT_STRICTNESS,

  /** This option deals with how to go about executing the build. */
  EXECUTION_STRATEGY,

  /** This option deals with build time optimizations. */
  BUILD_TIME_OPTIMIZATION,

  /**
   * This option lets a user specify WHICH output they wish the command to have. It might be a
   * selective filter on the outputs, or a blanket on/off switch.
   */
  OUTPUT_SELECTION,

  /**
   * This option lets a user configure the outputs. Unlike OUTPUT_SELECTION, which specifies whether
   * or not an output is built, this specifies qualities of the output.
   */
  OUTPUT_PARAMETERS,

  /**
   * This option provides information about signing outputs of the build. (For example, signing an
   * iOS application with a certificate.)
   */
  SIGNING,

  /**
   * This option affects semantics of the Starlark language or the build API accessible to BUILD
   * files, .bzl files, or WORKSPACE files.
   */
  STARLARK_SEMANTICS,

  /** This option dictates information about the test environment or test runner. */
  TESTING,

  /**
   * This option lets a user configure the toolchain used to execute actions in the build. This is
   * not to be used for parameters to a toolchain, which are more likely to fall into another
   * category; options in this category are for selecting between available toolchains, for example
   * based on execution-environment requirements.
   */
  TOOLCHAIN,

  /** This option relates to query output and semantics. */
  QUERY,

  /** This option relates to the `mod` subcommand. */
  MOD_COMMAND,

  /** This option relates to Bzlmod (external dependencies) output and semantics. */
  BZLMOD,

  /**
   * This option specifies or alters a generic input to a Bazel command. This category should only
   * be used if the input is generic and does not fall into other categories, such as toolchain-
   * specific inputs.
   */
  GENERIC_INPUTS,

  /** A category of options to configure Bazel's remote caching and execution capabilities. */
  REMOTE,
}
