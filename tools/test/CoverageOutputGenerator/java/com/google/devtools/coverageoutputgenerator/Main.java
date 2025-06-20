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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.devtools.coverageoutputgenerator.Constants.GCOV_EXTENSION;
import static com.google.devtools.coverageoutputgenerator.Constants.GCOV_JSON_EXTENSION;
import static com.google.devtools.coverageoutputgenerator.Constants.PROFDATA_EXTENSION;
import static com.google.devtools.coverageoutputgenerator.Constants.TRACEFILE_EXTENSION;
import static java.lang.Math.max;
import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/** Command line utility to convert raw coverage files to lcov (text) format. */
public class Main {
  private static final Logger logger = Logger.getLogger(Main.class.getName());

  public static void main(String... args) {
    try {
      int exitCode = runWithArgs(args);
      System.exit(exitCode);
    } catch (Exception e) {
      logger.log(Level.SEVERE, "Unhandled exception on lcov tool: " + e.getMessage(), e);
      System.exit(1);
    }
  }

  static int runWithArgs(String... args) throws ExecutionException, InterruptedException {
    LcovMergerFlags flags;
    try {
      flags = LcovMergerFlags.parseFlags(args);
    } catch (IllegalArgumentException e) {
      logger.log(Level.SEVERE, e.getMessage());
      return 1;
    }

    Path outputFile = Paths.get(flags.outputFile());

    List<Path> filesInCoverageDir =
        flags.coverageDir() != null
            ? getCoverageFilesInDir(Paths.get(flags.coverageDir()))
            : ImmutableList.of();
    Coverage coverage =
        Coverage.merge(
            parseFiles(
                getTracefiles(flags, filesInCoverageDir),
                LcovParser::parse,
                flags.parseParallelism()),
            parseFiles(
                getGcovInfoFiles(filesInCoverageDir), GcovParser::parse, flags.parseParallelism()),
            parseFiles(
                getGcovJsonInfoFiles(filesInCoverageDir),
                GcovJsonParser::parse,
                flags.parseParallelism()));

    if (flags.sourcesToReplaceFile() != null) {
      coverage.maybeReplaceSourceFileNames(getMapFromFile(flags.sourcesToReplaceFile()));
    }

    Path profdataFile = getProfdataFileOrNull(filesInCoverageDir);
    if (coverage.isEmpty()) {
      if (profdataFile == null) {
        try {
          logger.log(Level.WARNING, "There was no coverage found.");
          if (!Files.exists(outputFile)) {
            Files.createFile(outputFile); // Generate empty declared output
          }
        } catch (IOException e) {
          logger.log(
              Level.SEVERE,
              "Could not create empty output file " + outputFile + " due to: " + e.getMessage());
          return 1;
        }
      } else {
        // Bazel doesn't support yet converting profdata files to lcov. We still want to output a
        // coverage report so we copy the content of the profdata file to the output file. This is
        // not ideal but it unblocks some Bazel C++
        // coverage users.
        // TODO(#5881): Add support for profdata files.
        try {
          Files.copy(profdataFile, outputFile, REPLACE_EXISTING);
        } catch (IOException e) {
          logger.log(
              Level.SEVERE,
              "Could not copy file "
                  + profdataFile
                  + " to output file "
                  + outputFile
                  + " due to: "
                  + e.getMessage());
          return 1;
        }
      }
      return 0;
    }

    if (!coverage.isEmpty() && profdataFile != null) {
      // If there is one profdata file then there can't be other types of reports because there is
      // no way to merge them.
      // TODO(#5881): Add support for profdata files.
      logger.log(
          Level.WARNING,
          "Bazel doesn't support LLVM profdata coverage amongst other coverage formats.");
      return 0;
    }

    if (!flags.filterSources().isEmpty()) {
      coverage =
          Coverage.filterOutMatchingSources(
              coverage,
              flags.filterSources().stream().map(Pattern::compile).collect(toImmutableList()));
    }

    if (flags.hasSourceFileManifest()) {
      coverage =
          Coverage.getOnlyTheseSources(
              coverage, getSourcesFromSourceFileManifest(flags.sourceFileManifest()));
    }

    if (coverage.isEmpty()) {
      try {
        logger.log(Level.WARNING, "There was no coverage found.");
        if (!Files.exists(outputFile)) {
          Files.createFile(outputFile); // Generate empty declared output
        }
        return 0;
      } catch (IOException e) {
        logger.log(
            Level.SEVERE,
            "Could not create empty output file " + outputFile + " due to: " + e.getMessage());
        return 1;
      }
    }

    try {
      LcovPrinter.print(Files.newOutputStream(outputFile), coverage);
    } catch (IOException e) {
      logger.log(
          Level.SEVERE,
          "Could not write to output file " + outputFile + " due to " + e.getMessage());
      return 1;
    }
    return 0;
  }

  /**
   * Returns a set of source file names from the given manifest.
   *
   * <p>The manifest contains file names line by line. Each file can either be a source file (e.g.
   * .java, .cc) or a coverage metadata file (e.g. .gcno, .em).
   *
   * <p>This method only returns the C++ source files, ignoring the other files as they are not
   * necessary when putting together the final coverage report.
   */
  private static Set<String> getSourcesFromSourceFileManifest(String sourceFileManifest) {
    try (BufferedReader reader = Files.newBufferedReader(Path.of(sourceFileManifest))) {
      return reader.lines().filter(line -> !isMetadataFile(line)).collect(toImmutableSet());
    } catch (IOException e) {
      logger.log(Level.SEVERE, "Error reading file " + sourceFileManifest + ": " + e.getMessage());
      System.exit(1);
      throw new IllegalStateException("not reached");
    }
  }

  private static boolean isMetadataFile(String filename) {
    return filename.endsWith(".gcno") || filename.endsWith(".em");
  }

  private static List<Path> getGcovInfoFiles(List<Path> filesInCoverageDir) {
    List<Path> gcovFiles = getFilesWithExtension(filesInCoverageDir, GCOV_EXTENSION);
    if (gcovFiles.isEmpty()) {
      logger.log(Level.FINE, "No gcov info file found.");
    } else {
      logger.log(Level.FINE, "Found " + gcovFiles.size() + " gcov info files.");
    }
    return gcovFiles;
  }

  private static List<Path> getGcovJsonInfoFiles(List<Path> filesInCoverageDir) {
    List<Path> gcovJsonFiles = getFilesWithExtension(filesInCoverageDir, GCOV_JSON_EXTENSION);
    if (gcovJsonFiles.isEmpty()) {
      logger.log(Level.FINE, "No gcov json file found.");
    } else {
      logger.log(Level.FINE, "Found " + gcovJsonFiles.size() + " gcov json files.");
    }
    return gcovJsonFiles;
  }

  /**
   * Returns a .profdata file from the given files or null if none or more profdata files were
   * found.
   */
  private static Path getProfdataFileOrNull(List<Path> files) {
    List<Path> profdataFiles = getFilesWithExtension(files, PROFDATA_EXTENSION);
    if (profdataFiles.isEmpty()) {
      logger.log(Level.FINE, "No .profdata file found.");
      return null;
    }
    if (profdataFiles.size() > 1) {
      logger.log(
          Level.SEVERE,
          "Bazel currently supports only one profdata file per test. "
              + profdataFiles.size()
              + " .profadata files were found instead.");
      return null;
    }
    logger.log(Level.FINE, "Found one .profdata file.");
    return profdataFiles.get(0);
  }

  private static List<Path> getTracefiles(LcovMergerFlags flags, List<Path> filesInCoverageDir) {
    List<Path> lcovTracefiles;
    if (flags.reportsFile() != null) {
      lcovTracefiles = getTracefilesFromFile(Paths.get(flags.reportsFile()));
    } else if (flags.coverageDir() != null) {
      lcovTracefiles = getFilesWithExtension(filesInCoverageDir, TRACEFILE_EXTENSION);
    } else {
      lcovTracefiles = ImmutableList.of();
    }
    if (lcovTracefiles.isEmpty()) {
      logger.log(Level.FINE, "No lcov file found.");
    } else {
      logger.log(Level.FINE, "Found " + lcovTracefiles.size() + " tracefiles.");
    }
    return lcovTracefiles;
  }

  /**
   * Reads the content of the given file and returns a matching map.
   *
   * <p>It assumes the file contains lines in the form key:value. For each line it creates an entry
   * in the map with the corresponding key and value.
   */
  private static ImmutableMap<String, String> getMapFromFile(String file) {
    try (BufferedReader reader = Files.newBufferedReader(Path.of(file))) {
      return reader
          .lines()
          .map(line -> line.split(":"))
          .filter(entry -> entry.length == 2)
          .collect(toImmutableMap(entry -> entry[0], entry -> entry[1]));
    } catch (IOException e) {
      logger.log(Level.SEVERE, "Error reading file " + file + ": " + e.getMessage());
      System.exit(1);
      throw new IllegalStateException("not reached");
    }
  }

  static Coverage parseFiles(List<Path> files, Parser parser, int parallelism)
      throws ExecutionException, InterruptedException {
    if (parallelism == 1) {
      return parseFilesSequentially(files, parser);
    } else {
      return parseFilesInParallel(files, parser, parallelism);
    }
  }

  static Coverage parseFilesSequentially(List<Path> files, Parser parser) {
    Coverage coverage = new Coverage();
    for (Path file : files) {
      try {
        logger.log(Level.FINE, "Parsing file " + file);
        List<SourceFileCoverage> sourceFilesCoverage = parser.parse(Files.newInputStream(file));
        for (SourceFileCoverage sourceFileCoverage : sourceFilesCoverage) {
          coverage.add(sourceFileCoverage);
        }
      } catch (IOException e) {
        logger.log(
            Level.SEVERE,
            "File " + file.toAbsolutePath() + " could not be parsed due to: " + e.getMessage(),
            e);
        System.exit(1);
      }
    }
    return coverage;
  }

  static Coverage parseFilesInParallel(List<Path> files, Parser parser, int parallelism)
      throws ExecutionException, InterruptedException {
    try (ForkJoinPool pool = new ForkJoinPool(parallelism)) {
      int partitionSize = max(1, files.size() / parallelism);
      List<List<Path>> partitions = Lists.partition(files, partitionSize);
      return pool.submit(
              () ->
                  partitions.parallelStream()
                      .map((p) -> parseFilesSequentially(p, parser))
                      .reduce(Coverage::merge)
                      .orElse(Coverage.create()))
          .get();
    }
  }

  /**
   * Returns a list of all the files with the given extension found recursively under the given dir.
   */
  @VisibleForTesting
  static List<Path> getCoverageFilesInDir(Path dir) {
    try (Stream<Path> stream = Files.walk(dir)) {
      return stream
          .filter(
              p ->
                  p.toString().endsWith(TRACEFILE_EXTENSION)
                      || p.toString().endsWith(GCOV_EXTENSION)
                      || p.toString().endsWith(GCOV_JSON_EXTENSION)
                      || p.toString().endsWith(PROFDATA_EXTENSION))
          .collect(toImmutableList());
    } catch (IOException ex) {
      logger.log(Level.SEVERE, "Error reading folder " + dir + ": " + ex.getMessage());
      System.exit(1);
      throw new IllegalStateException("not reached");
    }
  }

  static List<Path> getFilesWithExtension(List<Path> files, String extension) {
    return files.stream()
        .filter(file -> file.toString().endsWith(extension))
        .collect(toImmutableList());
  }

  static List<Path> getTracefilesFromFile(Path reportsFile) {
    try (BufferedReader reader = Files.newBufferedReader(reportsFile)) {
      return reader.lines().map(Paths::get).collect(toImmutableList());
    } catch (IOException e) {
      logger.log(Level.SEVERE, "Error reading file " + reportsFile + ": " + e.getMessage());
      System.exit(1);
      throw new IllegalStateException("not reached");
    }
  }
}
