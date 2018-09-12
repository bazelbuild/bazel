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

package com.google.devtools.coverageoutputgenerator;

import static com.google.devtools.coverageoutputgenerator.Constants.GCOV_EXTENSION;
import static com.google.devtools.coverageoutputgenerator.Constants.TRACEFILE_EXTENSION;
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
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/** Command line utility to convert raw coverage files to lcov (text) format. */
public class Main {
  private static final Logger logger = Logger.getLogger(Main.class.getName());

  public static void main(String[] args) {
    LcovMergerFlags flags = null;
    try {
      flags = LcovMergerFlags.parseFlags(args);
    } catch (IllegalArgumentException e) {
      logger.log(Level.SEVERE, e.getMessage());
      System.exit(1);
    }

    List<File> filesInCoverageDir =
        flags.coverageDir() != null
            ? getCoverageFilesInDir(flags.coverageDir())
            : Collections.emptyList();
    Coverage coverage =
        Coverage.merge(
            parseFiles(getTracefiles(flags, filesInCoverageDir), LcovParser::parse),
            parseFiles(getGcovInfoFiles(filesInCoverageDir), GcovParser::parse));

    if (coverage.isEmpty()) {
      logger.log(Level.SEVERE, "There was no coverage found.");
      System.exit(1);
    }

    if (!flags.filterSources().isEmpty()) {
      coverage = Coverage.filterOutMatchingSources(coverage, flags.filterSources());
    }

    int exitStatus = 0;
    String outputFile = flags.outputFile();
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
      logger.log(Level.SEVERE, "No gcov info file found.");
    } else {
      logger.log(Level.INFO, "Found " + gcovFiles.size() + " gcov info files.");
    }
    return gcovFiles;
  }

  private static List<File> getTracefiles(LcovMergerFlags flags, List<File> filesInCoverageDir) {
    List<File> lcovTracefiles = new ArrayList<>();
    if (flags.coverageDir() != null) {
      lcovTracefiles = getFilesWithExtension(filesInCoverageDir, TRACEFILE_EXTENSION);
    } else if (flags.reportsFile() != null) {
      lcovTracefiles = getTracefilesFromFile(flags.reportsFile());
    }
    if (lcovTracefiles.isEmpty()) {
      logger.log(Level.SEVERE, "No lcov file found.");
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
    return files.stream()
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
}
