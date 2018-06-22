// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.lcovmerger;

import static com.google.devtools.lcovmerger.LcovConstants.TRACEFILE_EXTENSION;

import com.google.common.annotations.VisibleForTesting;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Command line utility to convert raw coverage files to lcov (text) format.
 */
public class Main {
  private static final Logger logger = Logger.getLogger(Main.class.getName());

  public static void main(String[] args) {
    Map<String, String> flags = null;
    try {
      flags = parseFlags(args);
    } catch (IllegalArgumentException e) {
      logger.log(Level.SEVERE, e.getMessage());
      System.exit(1);
    }

    List<File> lcovTracefiles = getLcovTracefiles(flags.get("coverage_dir"));
    if (lcovTracefiles.isEmpty()) {
      logger.log(Level.SEVERE, "No lcov file found.");
      System.exit(1);
    }
    Coverage coverage = new Coverage();
    for (File tracefile : lcovTracefiles) {
      try {
        List<SourceFileCoverage> sourceFilesCoverage =
            LcovParser.parse(new FileInputStream(tracefile));
        for (SourceFileCoverage sourceFileCoverage : sourceFilesCoverage) {
          coverage.add(sourceFileCoverage);
        }
      } catch (IOException e) {
        logger.log(Level.SEVERE, "Tracefile " + tracefile.getAbsolutePath() + " was deleted");
        System.exit(1);
      }
    }
    int exitStatus = 0;
    String outputFile = flags.get("output_file");
    try {
      File coverageFile = new File(outputFile);
      LcovPrinter.print(new FileOutputStream(coverageFile), coverage);
    } catch (IOException e) {
      logger.log(Level.SEVERE, "Could not write to output file " + outputFile);
      exitStatus = 1;
    }
    System.exit(exitStatus);
  }

  /**
   * Returns a list of all the files with a “.dat” extension found recursively under the given
   * directory.
   */
  @VisibleForTesting
  static List<File> getLcovTracefiles(String coverageDir) {
    List<File> datFiles = new ArrayList<>();
    try (Stream<Path> stream = Files.walk(Paths.get(coverageDir))) {
      datFiles = stream.filter(p -> p.toString().endsWith(TRACEFILE_EXTENSION))
          .map(path -> path.toFile())
          .collect(Collectors.toList());
    } catch (IOException ex) {
      logger.log(Level.SEVERE, "error reading folder " + coverageDir + ": " + ex.getMessage());
    }

    return datFiles;
  }

  /**
   * Parse flags in the form of "--coverage_dir=... -output_file=..."
   */
  private static Map<String, String> parseFlags(String[] args) {
    Map<String, String> flags = new HashMap<>();

    for (String arg : args) {
      if (!arg.startsWith("--")) {
        throw new IllegalArgumentException("Argument (" + arg + ") should start with --");
      }
      String[] parts = arg.substring(2).split("=", 2);
      if (parts.length != 2) {
        throw new IllegalArgumentException("There should be = in argument (" + arg + ")");
      }
      flags.put(parts[0], parts[1]);
    }

    // Validate flags
    if (!flags.containsKey("coverage_dir")) {
      throw new IllegalArgumentException("coverage_dir was not specified");
    }
    if (!flags.containsKey("output_file")) {
      // Different from blaze, this should be mandatory
      throw new IllegalArgumentException("output_file was not specified");
    }

    return flags;
  }
}
