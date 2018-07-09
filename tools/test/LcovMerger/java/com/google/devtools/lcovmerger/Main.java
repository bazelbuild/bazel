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
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
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

    List<File> lcovTracefiles = new ArrayList<>();
    if (flags.containsKey("coverage_dir")) {
      logger.log(Level.SEVERE, "Retrieving tracefiles from coverage_dir.");
      lcovTracefiles = getLcovTracefilesFromDir(flags.get("coverage_dir"));
    } else if (flags.containsKey("reports_file")) {
      logger.log(Level.SEVERE, "Retrieving tracefiles from reports_file.");
      lcovTracefiles = getLcovTracefilesFromFile(flags.get("reports_file"));
    }
    if (lcovTracefiles.isEmpty()) {
      logger.log(Level.SEVERE, "No lcov file found.");
      System.exit(1);
    }
    logger.log(Level.SEVERE, "Found " + lcovTracefiles.size() + " tracefiles.");
    Coverage coverage = new Coverage();
    for (File tracefile : lcovTracefiles) {
      try {
        logger.log(Level.SEVERE, "Parsing tracefile " + tracefile.toString());
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
      logger.log(Level.SEVERE,
          "Could not write to output file " + outputFile + " due to " + e.getMessage());
      exitStatus = 1;
    }
    System.exit(exitStatus);
  }

  /**
   * Returns a list of all the files with a “.dat” extension found recursively under the given
   * directory.
   */
  @VisibleForTesting
  static List<File> getLcovTracefilesFromDir(String coverageDir) {
    List<File> datFiles = new ArrayList<>();
    try (Stream<Path> stream = Files.walk(Paths.get(coverageDir))) {
      datFiles = stream.filter(p -> p.toString().endsWith(TRACEFILE_EXTENSION))
          .map(path -> path.toFile())
          .collect(Collectors.toList());
    } catch (IOException ex) {
      logger.log(Level.SEVERE, "Error reading folder " + coverageDir + ": " + ex.getMessage());
    }

    return datFiles;
  }

  static List<File> getLcovTracefilesFromFile(String reportsFile) {
    List<File> datFiles = new ArrayList<>();
    try (FileInputStream inputStream = new FileInputStream(reportsFile)) {
      InputStreamReader inputStreamReader = new InputStreamReader(inputStream, UTF_8);
      BufferedReader reader = new BufferedReader(inputStreamReader);
      for (String tracefile = reader.readLine(); tracefile != null; tracefile = reader.readLine()) {
        // TODO(elenairina): baseline coverage contains some file names that need to be modified
        if (!tracefile.endsWith("baseline_coverage.dat")) {
          datFiles.add(new File(tracefile));
        }
      }

    } catch (IOException e) {
      logger.log(Level.SEVERE, "Error reading file " + reportsFile + ": " + e.getMessage());
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
    for (String flag : flags.keySet()) {
      switch (flag) {
        case "coverage_dir":
        case "reports_file":
        case "output_file":
          continue;
        default:
          throw new IllegalArgumentException("Unknown flag --" + flag);
      }
    }

    if (!flags.containsKey("coverage_dir") && !flags.containsKey("reports_file")) {
      throw new IllegalArgumentException(
          "At least one of --coverage_dir or --reports_file should be specified.");
    }
    if (flags.containsKey("coverage_dir") && flags.containsKey("reports_file")) {
      throw new IllegalArgumentException(
          "Only one of --coverage_dir or --reports_file must be specified.");
    }
    if (!flags.containsKey("output_file")) {
      // Different from blaze, this should be mandatory
      throw new IllegalArgumentException("--output_file was not specified");
    }

    return flags;
  }
}
