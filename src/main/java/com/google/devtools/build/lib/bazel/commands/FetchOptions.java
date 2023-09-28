// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.commands;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;

/** Defines the options specific to Bazel's sync command */
public class FetchOptions extends OptionsBase {
  @Option(
      name = "all",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      effectTags = {OptionEffectTag.CHANGES_INPUTS},
      help =
          "Fetches all external repositories necessary for building any target or repository. Only"
              + " works when --enable_bzlmod is on.")
  public boolean all;

  @Option(
      name = "configure",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      effectTags = {OptionEffectTag.CHANGES_INPUTS},
      help =
          "Only fetch repositories marked as 'configure' for system-configuration purpose. Only"
              + " works when --enable_bzlmod is on.")
  public boolean configure;

  /*TODO(salmasamy) add more options:
   * repo: to fetch a specific repo
   * force: to force fetch even if a repo exists
   */
}
