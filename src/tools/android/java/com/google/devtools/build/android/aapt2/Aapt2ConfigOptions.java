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
// Copyright 2017 The Bazel Authors. All rights reserved.

package com.google.devtools.build.android.aapt2;

import com.android.repository.Revision;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.RevisionConverter;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.TriState;
import java.nio.file.Path;
import java.util.List;

/** Aapt2 specific configuration options. */
public class Aapt2ConfigOptions extends OptionsBase {
  @Option(
      name = "aapt2",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      defaultValue = "null",
      converter = ExistingPathConverter.class,
      category = "tool",
      help = "Aapt2 tool location for resource compilation.")
  public Path aapt2;

  @Option(
      name = "buildToolsVersion",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      defaultValue = "null",
      converter = RevisionConverter.class,
      category = "config",
      help = "Version of the build tools (e.g. aapt) being used, e.g. 23.0.2")
  public Revision buildToolsVersion;

  @Option(
      name = "androidJar",
      defaultValue = "null",
      converter = ExistingPathConverter.class,
      category = "tool",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Path to the android jar for resource packaging and building apks.")
  public Path androidJar;

  @Option(
      name = "useAaptCruncher",
      defaultValue = "auto",
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Use the legacy aapt cruncher, defaults to true for non-LIBRARY packageTypes. "
              + " LIBRARY packages do not benefit from the additional processing as the resources"
              + " will need to be reprocessed during the generation of the final apk. See"
              + " https://code.google.com/p/android/issues/detail?id=67525 for a discussion of the"
              + " different png crunching methods.")
  public TriState useAaptCruncher;

  @Option(
      name = "conditionalKeepRules",
      defaultValue = "auto",
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Have AAPT2 produce conditional keep rules.")
  public TriState conditionalKeepRules;

  @Option(
      name = "uncompressedExtensions",
      defaultValue = "",
      converter = CommaSeparatedOptionListConverter.class,
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "A list of file extensions not to compress.")
  public List<String> uncompressedExtensions;

  @Option(
      name = "debug",
      defaultValue = "false",
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Indicates if it is a debug build.")
  public boolean debug;

  @Option(
      name = "resourceConfigs",
      defaultValue = "",
      converter = CommaSeparatedOptionListConverter.class,
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "A list of resource config filters to pass to aapt.")
  public List<String> resourceConfigs;

  private static final String ANDROID_SPLIT_DOCUMENTATION_URL =
      "https://developer.android.com/guide/topics/resources/providing-resources.html"
          + "#QualifierRules";

  @Option(
      name = "split",
      defaultValue = "null",
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      allowMultiple = true,
      help =
          "An individual split configuration to pass to aapt."
              + " Each split is a list of configuration filters separated by commas."
              + " Configuration filters are lists of configuration qualifiers separated by dashes,"
              + " as used in resource directory names and described on the Android developer site: "
              + ANDROID_SPLIT_DOCUMENTATION_URL
              + " For example, a split might be 'en-television,en-xxhdpi', containing English"
              + " assets which either are for TV screens or are extra extra high resolution."
              + " Multiple splits can be specified by passing this flag multiple times."
              + " Each split flag will produce an additional output file, named by replacing the"
              + " commas in the split specification with underscores, and appending the result to"
              + " the output package name following an underscore.")
  public List<String> splits;

  // TODO(b/136572475, b/112848607): remove this option
  @Option(
      name = "useCompiledResourcesForMerge",
      defaultValue = "true",
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Use compiled resources for merging rather than parsed symbols binary.",
      deprecationWarning = "cannot be disabled")
  public boolean useCompiledResourcesForMerge;

  @Option(
      name = "resourceTableAsProto",
      defaultValue = "false",
      category = "config",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Generate the resource table as a protocol buffer.")
  public boolean resourceTableAsProto;

  @Option(
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      name = "generatePseudoLocale",
      defaultValue = "true",
      category = "config",
      help = "Whether to generate pseudo locales during compilation.")
  public boolean generatePseudoLocale;
}
