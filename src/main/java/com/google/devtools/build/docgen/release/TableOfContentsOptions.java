// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.docgen.release;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;

/** Command line options for the TOC updater. */
public class TableOfContentsOptions extends OptionsBase {
  @Option(
      name = "input",
      abbrev = 'i',
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path to a YAML file that contains the old table of contents.")
  public String inputPath;

  @Option(
      name = "output",
      abbrev = 'o',
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path of the YAML file where the new TOC should be written to.")
  public String outputPath;

  @Option(
      name = "version_indicator_input",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path of the file containing the version indicator.")
  public String versionIndicatorInputPath;

  @Option(
      name = "version_indicator_output",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path of the file where the version indicator should be written.")
  public String versionIndicatorOutputPath;

  @Option(
      name = "version",
      abbrev = 'v',
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "The name of the Bazel release that should be included in the TOC.")
  public String version;

  @Option(
      name = "max_releases",
      abbrev = 'm',
      defaultValue = "5",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Maximum number of Bazel releases that should be included in the TOC.")
  public int maxReleases;

  @Option(
      name = "help",
      abbrev = 'h',
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Prints the help string.")
  public boolean printHelp;

  public boolean isValid() {
    return !inputPath.isEmpty() && !outputPath.isEmpty() && !version.isEmpty() && maxReleases > 0;
  }
}
