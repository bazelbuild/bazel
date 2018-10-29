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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import java.util.List;
import javax.annotation.Nullable;

@AutoValue
abstract class LcovMergerFlags {

  @Nullable
  abstract String coverageDir();

  @Nullable
  abstract String reportsFile();

  abstract String outputFile();

  abstract List<String> filterSources();

  /**
   * The path to a source file manifest. This file contains multiple lines that represent file names
   * of the sources that the final coverage report must include. Additionally this file can also
   * contain coverage metadata files (e.g. gcno, .em), which can be ignored.
   *
   * @return
   */
  @Nullable
  abstract String sourceFileManifest();

  @Nullable
  abstract String sourcesToReplaceFile();

  /** Parse flags in the form of "--coverage_dir=... -output_file=..." */
  static LcovMergerFlags parseFlags(String[] args) {
    ImmutableList.Builder<String> filterSources = new ImmutableList.Builder<>();
    String coverageDir = null;
    String reportsFile = null;
    String outputFile = null;
    String sourceFileManifest = null;
    String sourcesToReplaceFile = null;

    for (String arg : args) {
      if (!arg.startsWith("--")) {
        throw new IllegalArgumentException("Argument (" + arg + ") should start with --");
      }
      String[] parts = arg.substring(2).split("=", 2);
      if (parts.length != 2) {
        throw new IllegalArgumentException("There should be = in argument (" + arg + ")");
      }
      switch (parts[0]) {
        case "coverage_dir":
          coverageDir = parts[1];
          break;
        case "reports_file":
          reportsFile = parts[1];
          break;
        case "output_file":
          outputFile = parts[1];
          break;
        case "filter_sources":
          filterSources.add(parts[1]);
          break;
        case "source_file_manifest":
          sourceFileManifest = parts[1];
          break;
        case "sources_to_replace_file":
          sourcesToReplaceFile = parts[1];
          break;
        default:
          throw new IllegalArgumentException("Unknown flag " + arg);
      }
    }

    if (coverageDir == null && reportsFile == null) {
      throw new IllegalArgumentException(
          "At least one of --coverage_dir or --reports_file should be specified.");
    }
    if (coverageDir != null && reportsFile != null) {
      throw new IllegalArgumentException(
          "Only one of --coverage_dir or --reports_file must be specified.");
    }
    if (outputFile == null) {
      throw new IllegalArgumentException("--output_file was not specified.");
    }
    return new AutoValue_LcovMergerFlags(
        coverageDir,
        reportsFile,
        outputFile,
        filterSources.build(),
        sourceFileManifest,
        sourcesToReplaceFile);
  }

  boolean hasSourceFileManifest() {
    return sourceFileManifest() != null;
  }
}
