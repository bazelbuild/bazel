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

package com.google.devtools.build.skydoc;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.util.List;

/** Contains options for running {@link SkydocMain}. */
public class SkydocOptions extends OptionsBase {

  @Option(
      name = "input",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = OptionEffectTag.UNKNOWN,
      help = "The label of the target file for which to generate documentation")
  public String targetFileLabel;

  @Option(
      name = "workspace_name",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = OptionEffectTag.UNKNOWN,
      help = "The name of the workspace in which the input file resides")
  public String workspaceName;

  @Option(
      name = "output",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = OptionEffectTag.UNKNOWN,
      help = "The path of the file to output documentation into")
  public String outputFilePath;

  @Option(
      name = "symbols",
      allowMultiple = true,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = OptionEffectTag.UNKNOWN,
      help =
          "A list of symbol names to generate documentation for. These should correspond to the"
              + " names of rule, provider, or function definitions in the input file. If this list"
              + " is empty, then documentation for all exported rule definitions will be"
              + " generated.")
  public List<String> symbolNames;

  @Option(
      name = "dep_roots",
      allowMultiple = true,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = OptionEffectTag.UNKNOWN,
      help = "File path roots to search when resolving transitive bzl dependencies")
  public List<String> depRoots;
}
