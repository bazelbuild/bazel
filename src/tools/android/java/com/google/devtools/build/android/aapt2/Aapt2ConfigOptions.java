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
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.Converters.CompatExistingPathConverter;
import com.google.devtools.build.android.Converters.CompatRevisionConverter;
import com.google.devtools.build.android.TriState;
import java.nio.file.Path;
import java.util.List;

/** Aapt2 specific configuration options. */
@Parameters(separators = "= ")
public class Aapt2ConfigOptions {
  @Parameter(
      names = "--aapt2",
      converter = CompatExistingPathConverter.class,
      description = "Aapt2 tool location for resource compilation.")
  public Path aapt2;

  @Parameter(
      names = "--buildToolsVersion",
      converter = CompatRevisionConverter.class,
      description = "Version of the build tools (e.g. aapt) being used, e.g. 23.0.2")
  public Revision buildToolsVersion;

  @Parameter(
      names = "--androidJar",
      converter = CompatExistingPathConverter.class,
      description = "Path to the android jar for resource packaging and building apks.")
  public Path androidJar;

  @Parameter(
      names = "--useAaptCruncher",
      description =
          "Use the legacy aapt cruncher, defaults to true for non-LIBRARY packageTypes. "
              + " LIBRARY packages do not benefit from the additional processing as the resources"
              + " will need to be reprocessed during the generation of the final apk. See"
              + " https://code.google.com/p/android/issues/detail?id=67525 for a discussion of the"
              + " different png crunching methods.")
  public TriState useAaptCruncher = TriState.AUTO;

  @Parameter(
      names = "--conditionalKeepRules",
      description = "Have AAPT2 produce conditional keep rules.")
  public TriState conditionalKeepRules = TriState.AUTO;

  @Parameter(
      names = "--uncompressedExtensions",
      description = "A list of file extensions not to compress.")
  public List<String> uncompressedExtensions = ImmutableList.of();

  @Parameter(names = "--debug", description = "Indicates if it is a debug build.", arity = 1)
  public boolean debug;

  @Parameter(
      names = "--resourceConfigs",
      description = "A list of resource config filters to pass to aapt.")
  public List<String> resourceConfigs = ImmutableList.of();

  private static final String ANDROID_SPLIT_DOCUMENTATION_URL =
      "https://developer.android.com/guide/topics/resources/providing-resources.html"
          + "#QualifierRules";

  @Parameter(
      names = "--split",
      description =
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
  public List<String> splits = ImmutableList.of();

  // TODO(b/136572475, b/112848607): remove this option
  @Parameter(
      names = "--useCompiledResourcesForMerge",
      arity = 1,
      description = "Use compiled resources for merging rather than parsed symbols binary.")
  public boolean useCompiledResourcesForMerge = true;

  @Parameter(
      names = "--resourceTableAsProto",
      arity = 1,
      description = "Generate the resource table as a protocol buffer.")
  public boolean resourceTableAsProto;

  @Parameter(
      names = "--generatePseudoLocale",
      arity = 1,
      description = "Whether to generate pseudo locales during compilation.")
  public boolean generatePseudoLocale = true;

  @Parameter(
      names = "--useDataBindingAndroidX",
      arity = 1,
      description = "Indicates whether databinding generated files should depend on AndroidX.")
  public boolean useDataBindingAndroidX;
}
