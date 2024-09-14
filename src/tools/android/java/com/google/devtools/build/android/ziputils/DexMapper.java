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

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.AndroidOptionsUtils;
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
    DexMapperOptions options = new DexMapperOptions();
    String[] preprocessedArgs = AndroidOptionsUtils.runArgFilePreprocessor(args);
    String[] normalizedArgs =
        AndroidOptionsUtils.normalizeBooleanOptions(options, preprocessedArgs);
    JCommander.newBuilder().addObject(options).build().parse(normalizedArgs);

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
  @Parameters(separators = "= ")
  public static class DexMapperOptions {
    @Parameter(
        names = {"--input_jar", "-i"},
        description =
            "Input file to read classes and jars from. Classes in "
                + " earlier files override those in later ones.")
    public List<String> inputJars = ImmutableList.of();

    @Parameter(
        names = {"--output_jar", "-o"},
        description =
            "Output file to write. Each argument is one shard. "
                + "Output files are filled in the order specified.")
    public List<String> outputJars = ImmutableList.of();

    @Parameter(
        names = {"--main_dex_filter", "-f"},
        description = "List of classes to include in the first output file.")
    public String mainDexFilter;

    @Parameter(
        names = {"--output_resources", "-r"},
        description = "File to write the Java resources to.")
    public String outputResources;

    @Parameter(
        names = "--split_dexed_classes",
        arity = 1,
        description = "Split X.class.dex like X.class if true.  Treated as resources if false.")
    public boolean splitDexedClasses;

    @Parameter(
        names = "--inclusion_filter_jar",
        description =
            "Only copy entries that are listed in the given Jar file.  By default, all entries "
                + "are copied over.")
    public String inclusionFilterJar;
  }
}
