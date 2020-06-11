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

package com.google.devtools.coverageoutputgenerator;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import com.beust.jcommander.Parameters;
import com.google.common.collect.ImmutableList;
import java.util.List;
import java.util.logging.Logger;
import javax.annotation.Nullable;

@Parameters(separators = "= ", optionPrefixes = "--")
class LcovMergerFlags {
  private static final Logger logger = Logger.getLogger(LcovMergerFlags.class.getName());
  private static final int DEFAULT_PARSE_FILE_PARALLELISM = 8;

  @Parameter(names = "--coverage_dir")
  private String coverageDir;

  @Nullable
  @Parameter(names = {"--reports_file", "--lcovfile_path"})
  private String reportsFile;

  @Parameter(names = "--output_file")
  private String outputFile;

  @Parameter(names = "--filter_sources")
  private List<String> filterSources;

  /**
   * The path to a source file manifest. This file contains multiple lines that represent file names
   * of the sources that the final coverage report must include. Additionally this file can also
   * contain coverage metadata files (e.g. gcno, .em), which can be ignored.
   *
   * @return
   */
  @Nullable
  @Parameter(names = "--source_file_manifest")
  private String sourceFileManifest;

  @Nullable
  @Parameter(names = "--sources_to_replace_file")
  private String sourcesToReplaceFile;

  @Parameter(names = "--parse_parallelism")
  private Integer parseParallelism;

  public String coverageDir() {
    return coverageDir;
  }

  public String outputFile() {
    return outputFile;
  }

  public List<String> filterSources() {
    return filterSources == null ? ImmutableList.of() : filterSources;
  }

  public String reportsFile() {
    return reportsFile;
  }

  public String sourceFileManifest() {
    return sourceFileManifest;
  }

  public String sourcesToReplaceFile() {
    return sourcesToReplaceFile;
  }

  boolean hasSourceFileManifest() {
    return sourceFileManifest != null;
  }

  int parseParallelism() {
    return parseParallelism == null ? DEFAULT_PARSE_FILE_PARALLELISM : parseParallelism;
  }

  static LcovMergerFlags parseFlags(String[] args) {
    LcovMergerFlags flags = new LcovMergerFlags();
    JCommander jCommander = new JCommander(flags);
    jCommander.setAllowParameterOverwriting(true);
    jCommander.setAcceptUnknownOptions(true);
    try {
      jCommander.parse(args);
    } catch (ParameterException e) {
      throw new IllegalArgumentException("Error parsing args", e);
    }
    if (flags.coverageDir == null && flags.reportsFile == null) {
      throw new IllegalArgumentException(
          "At least one of coverage_dir or reports_file should be specified.");
    }
    if (flags.coverageDir != null && flags.reportsFile != null) {
      logger.warning("Overriding --coverage_dir value in favor of --reports_file");
    }
    if (flags.outputFile == null) {
      throw new IllegalArgumentException("output_file was not specified.");
    }
    return flags;
  }
}
