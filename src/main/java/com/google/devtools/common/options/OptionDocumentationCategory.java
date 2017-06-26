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
 * <p>Grouping things is only a useful exercise if the group includes multiple units. Please only
 * add categories that group a useful subset of options. (>2, <200, let's say)
 */
public enum OptionDocumentationCategory {
  /**
   * A category to aid transition, to make it obvious that an option needs to be categorized. Note:
   * Do NOT add this to new flags!
   */
  UNCATEGORIZED,

  /**
   * Startup options appear before the command and are parsed by the client. Changing them may cause
   * a server restart, see OptionEffectTag.LOSES_INCREMENTAL_STATE.
   */
  BAZEL_CLIENT_OPTIONS,

  /** This option's primary purpose is to affect the verbosity, format or location of logging. */
  LOGGING,

  /** This option deals with how to go about executing the build. */
  EXECUTION_STRATEGY,

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
}
