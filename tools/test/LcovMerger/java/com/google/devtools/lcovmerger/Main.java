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

import static com.google.devtools.lcovmerger.Constants.GCOV_EXTENSION;
import static com.google.devtools.lcovmerger.Constants.TRACEFILE_EXTENSION;
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
import java.util.Collections;
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

    List<File> filesInCoverageDir =
        flags.containsKey("coverage_dir")
            ? getCoverageFilesInDir(flags.get("coverage_dir"))
            : Collections.emptyList();
    Coverage coverage =
        Coverage.merge(
            parseFiles(getTracefiles(flags, filesInCoverageDir), LcovParser::parse),
            parseFiles(getGcovInfoFiles(filesInCoverageDir), GcovParser::parse));

    if (coverage.isEmpty()) {
      logger.log(Level.SEVERE, "There was no coverage found.");
      System.exit(1);
    }

    if (flags.containsKey("filter_out_regexes")) {
      String[] regexes = flags.get("filter_out_regexes").split(",");
      coverage = coverage.filterOutMatchingSources(coverage, regexes);
    }

    int exitStatus = 0;
    String outputFile = flags.get("output_file");
    try {
      LcovPrinter.print(new FileOutputStream(new File(outputFile)), coverage);
    } catch (IOException e) {
      logger.log(
          Level.SEVERE,
          "Could not write to output file " + outputFile + " due to " + e.getMessage());
      exitStatus = 1;
    }
    System.exit(exitStatus);
  }

  private static List<File> getGcovInfoFiles(List<File> filesInCoverageDir) {
    List<File> gcovFiles = getFilesWithExtension(filesInCoverageDir, GCOV_EXTENSION);
    if (gcovFiles.isEmpty()) {
      logger.log(Level.INFO, "No gcov info file found.");
    } else {
      logger.log(Level.INFO, "Found " + gcovFiles.size() + " gcov info files.");
    }
    return gcovFiles;
  }

  private static List<File> getTracefiles(
      Map<String, String> flags, List<File> filesInCoverageDir) {
    List<File> lcovTracefiles = new ArrayList<>();
    if (flags.containsKey("coverage_dir")) {
      lcovTracefiles = getFilesWithExtension(filesInCoverageDir, TRACEFILE_EXTENSION);
    } else if (flags.containsKey("reports_file")) {
      lcovTracefiles = getTracefilesFromFile(flags.get("reports_file"));
    }
    if (lcovTracefiles.isEmpty()) {
      logger.log(Level.INFO, "No lcov file found.");
    } else {
      logger.log(Level.INFO, "Found " + lcovTracefiles.size() + " tracefiles.");
    }
    return lcovTracefiles;
  }

  private static Coverage parseFiles(List<File> files, Parser parser) {
    Coverage coverage = new Coverage();
    for (File file : files) {
      try {
        logger.log(Level.SEVERE, "Parsing file " + file.toString());
        List<SourceFileCoverage> sourceFilesCoverage = parser.parse(new FileInputStream(file));
        for (SourceFileCoverage sourceFileCoverage : sourceFilesCoverage) {
          coverage.add(sourceFileCoverage);
        }
      } catch (IOException e) {
        logger.log(
            Level.SEVERE,
            "File " + file.getAbsolutePath() + " could not be parsed due to: " + e.getMessage());
        System.exit(1);
      }
    }
    return coverage;
  }

  /**
   * Returns a list of all the files with the given extension found recursively under the given dir.
   */
  @VisibleForTesting
  static List<File> getCoverageFilesInDir(String dir) {
    List<File> files = new ArrayList<>();
    try (Stream<Path> stream = Files.walk(Paths.get(dir))) {
      files =
          stream
              .filter(
                  p ->
                      p.toString().endsWith(TRACEFILE_EXTENSION)
                          || p.toString().endsWith(GCOV_EXTENSION))
              .map(path -> path.toFile())
              .collect(Collectors.toList());
    } catch (IOException ex) {
      logger.log(Level.SEVERE, "Error reading folder " + dir + ": " + ex.getMessage());
    }
    return files;
  }

  static List<File> getFilesWithExtension(List<File> files, String extension) {
    return files
        .stream()
        .filter(file -> file.toString().endsWith(extension))
        .collect(Collectors.toList());
  }

  static List<File> getTracefilesFromFile(String reportsFile) {
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
        case "filter_out_regexes":
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
