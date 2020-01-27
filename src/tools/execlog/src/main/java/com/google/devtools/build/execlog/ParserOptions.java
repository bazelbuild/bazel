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

package com.google.devtools.build.execlog;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.util.List;

/** Options for execution log parser. */
public class ParserOptions extends OptionsBase {
  @Option(
      name = "log_path",
      defaultValue = "null",
      category = "logging",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      allowMultiple = true,
      help =
          "Location of the log file to parse. If a second value is specified, both files will be"
              + " converted. The first log will be processed as-is. The actions in the second log"
              + " will be reordered to match the first. An action will be matched if the first"
              + " file contains an action with the same name of the first output. Any actions"
              + " that cannot be matched to the first file will appear at the end of the log. Note"
              + " that this reordering fascilitates easier text-based comparisons, but may break"
              + " any logical order of the actions.")
  public List<String> logPath;

  @Option(
      name = "output_path",
      defaultValue = "null",
      category = "logging",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      allowMultiple = true,
      help =
          "Location where to put the output(s). If left empty, the log will be output to stdout."
              + " If two log paths are specified, needs to be specified and have exactly two"
              + " paths.")
  public List<String> outputPath;

  @Option(
      name = "restrict_to_runner",
      defaultValue = "null",
      category = "logging",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "If set, only output the executions that used the given runner.")
  public String restrictToRunner;
}
