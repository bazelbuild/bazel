// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.workspacelog;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.util.List;

/** Options for workspace log parser. */
public class WorkspaceLogParserOptions extends OptionsBase {
  @Option(
      name = "log_path",
      defaultValue = "null",
      category = "logging",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Location of the workspace rules log file to parse.")
  public String logPath;

  @Option(
      name = "output_path",
      defaultValue = "null",
      category = "logging",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Location where to put the output. If left empty, the log will be output to stdout.")
  public String outputPath;

  @Option(
      name = "exclude_rule",
      defaultValue = "null",
      category = "logging",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.UNKNOWN},
      allowMultiple = true,
      help = "Rule(s) to filter out while parsing.")
  public List<String> excludeRule;
}
