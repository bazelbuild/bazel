// Copyright 2014 The Bazel Authors. All rights reserved.
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

/**
 * Command line options for the Build Encyclopedia docgen.
 */
public class BuildEncyclopediaOptions extends OptionsBase {
  @Option(
      name = "link_map_path",
      abbrev = 'm',
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Path to a JSON file that specifies link mappings (page name to URL and input file/label"
              + " to source code repository URL).")
  public String linkMapPath;

  @Option(
      name = "input_dir",
      abbrev = 'i',
      defaultValue = "null",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "An input directory to read Java source files")
  public List<String> inputJavaDirs;

  // The source tree root must be passed on the command line, instead of in link_map_path json
  // content, because its schema varies wildly depending on execution platform (e.g. RBE vs. local).
  @Option(
      name = "input_root",
      abbrev = 'r',
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Directory of the source tree root")
  public String inputRoot;

  @Option(
      name = "input_stardoc_proto",
      defaultValue = "null",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "A stardoc_output.ModuleInfo binary proto file generated from a Build Encyclopedia entry"
              + " point .bzl file; documentation from input_stardoc_proto takes precedence over"
              + " documentation from input_dir")
  public List<String> inputStardocProtos;

  @Option(
    name = "provider",
    abbrev = 'p',
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "The name of the rule class provider"
  )
  public String provider;

  @Option(
      name = "output_file",
      abbrev = 'f',
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "An output file.")
  public String outputFile;

  @Option(
    name = "output_dir",
    abbrev = 'o',
    defaultValue = ".",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "An output directory."
  )
  public String outputDir;

  @Option(
      name = "denylist",
      oldName = "blacklist",
      abbrev = 'b',
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "A path to a file listing rules not to document.")
  public String denylist;

  @Option(
    name = "single_page",
    abbrev = '1',
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Whether to generate the BE as a single HTML page or one page per rule family."
  )
  public boolean singlePage;

  @Option(
    name = "help",
    abbrev = 'h',
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Prints the help string."
  )
  public boolean help;

  @Option(
      name = "create_toc",
      abbrev = 't',
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Whether a table-of-contents file should be created. Mutually exclusive with"
              + " --single_page.")
  public boolean createToc;
}
