// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.ziputils;

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.nio.file.FileSystems;
import java.util.List;

/**
 * Command-line entry point for dex mapper utility, that maps an applications classes (given in
 * one or more input jar files), to one or more output jars, that may then be compiled separately.
 * This class performs command line parsing and delegates to an {@link SplitZip} instance.
 *
 * <p>The result of compiling each output archive can be consolidated using the dex reducer utility.
 */
public class DexMapper {

  /**
   * @param args the command line arguments
   */
  public static void main(String[] args) {

    OptionsParser parser =
        OptionsParser.builder()
            .optionsClasses(DexMapperOptions.class)
            .allowResidue(true)
            .argsPreProcessor(new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()))
            .build();
    parser.parseAndExitUponError(args);
    DexMapperOptions options = parser.getOptions(DexMapperOptions.class);

    List<String> inputs = options.inputJars;
    List<String> outputs = options.outputJars;
    String filterFile = options.mainDexFilter;
    String resourceFile = options.outputResources;

    try {
      // Always drop desugaring metadata, which we check elsewhere and don't want in final APKs
      // (see b/65645388).
      Predicate<String> inputFilter = Predicates.not(Predicates.equalTo("META-INF/desugar_deps"));
      if (options.inclusionFilterJar != null) {
        inputFilter =
            Predicates.and(inputFilter, SplitZipFilters.entriesIn(options.inclusionFilterJar));
      }
      new SplitZip()
          .setVerbose(false)
          .useDefaultEntryDate()
          .setSplitDexedClasses(options.splitDexedClasses)
          .addInputs(inputs)
          .addOutputs(outputs)
          .setInputFilter(inputFilter)
          .setMainClassListFile(filterFile)
          .setResourceFile(resourceFile)
          .run()
          .close();
    } catch (Exception ex) {
        System.err.println("Caught exception" + ex.getMessage());
        ex.printStackTrace(System.out);
        System.exit(1);
    }
  }

  /** Commandline options. */
  public static class DexMapperOptions extends OptionsBase {
    @Option(
      name = "input_jar",
      defaultValue = "null",
      category = "input",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      abbrev = 'i',
      help =
          "Input file to read classes and jars from. Classes in "
              + " earlier files override those in later ones."
    )
    public List<String> inputJars;

    @Option(
      name = "output_jar",
      defaultValue = "null",
      category = "output",
      allowMultiple = true,
      abbrev = 'o',
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Output file to write. Each argument is one shard. "
              + "Output files are filled in the order specified."
    )
    public List<String> outputJars;

    @Option(
      name = "main_dex_filter",
      defaultValue = "null",
      category = "input",
      abbrev = 'f',
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "List of classes to include in the first output file."
    )
    public String mainDexFilter;

    @Option(
      name = "output_resources",
      defaultValue = "null",
      category = "output",
      abbrev = 'r',
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "File to write the Java resources to."
    )
    public String outputResources;

    @Option(
      name = "split_dexed_classes",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Split X.class.dex like X.class if true.  Treated as resources if false."
    )
    public boolean splitDexedClasses;

    @Option(
      name = "inclusion_filter_jar",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Only copy entries that are listed in the given Jar file.  By default, all entries "
              + "are copied over."
    )
    public String inclusionFilterJar;
  }
}
