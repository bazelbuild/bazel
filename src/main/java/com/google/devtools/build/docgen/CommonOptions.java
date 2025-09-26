// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.docgen;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.util.List;

/** Command line options shared by Build Encyclopedia and Starlark and BUILD language API docgen. */
public class CommonOptions extends OptionsBase {
  @Option(
      name = "link_map_path",
      abbrev = 'm',
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Path to a JSON file that specifies link mappings (page name to URL and input file/label"
              + " to source code repository URL). Must be specified.")
  public String linkMapPath;

  @Option(
      name = "api_stardoc_proto",
      defaultValue = "null",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "A stardoc_output.ModuleInfo binary proto file generated from a Starlark and BUILD"
              + " language API entry point .bzl file")
  public List<String> apiStardocProtos;

  @Option(
      name = "create_toc",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Whether to output a table of contents.")
  public boolean createToc;

  @Option(
      name = "help",
      abbrev = 'h',
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Prints the help string.")
  public boolean help;
}
