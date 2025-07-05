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

import static com.google.devtools.coverageoutputgenerator.Constants.DELIMITER;
import static com.google.devtools.coverageoutputgenerator.Constants.GCOV_BRANCH_MARKER;
import static com.google.devtools.coverageoutputgenerator.Constants.GCOV_BRANCH_NOTEXEC;
import static com.google.devtools.coverageoutputgenerator.Constants.GCOV_BRANCH_NOTTAKEN;
import static com.google.devtools.coverageoutputgenerator.Constants.GCOV_BRANCH_TAKEN;
import static com.google.devtools.coverageoutputgenerator.Constants.GCOV_CWD_MARKER;
import static com.google.devtools.coverageoutputgenerator.Constants.GCOV_FILE_MARKER;
import static com.google.devtools.coverageoutputgenerator.Constants.GCOV_FUNCTION_MARKER;
import static com.google.devtools.coverageoutputgenerator.Constants.GCOV_LINE_MARKER;
import static com.google.devtools.coverageoutputgenerator.Constants.GCOV_VERSION_MARKER;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ListMultimap;
import com.google.common.collect.MultimapBuilder;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A {@link Parser} for gcov intermediate format. See the flag {@code --intermediate-format} in <a
 * href="https://gcc.gnu.org/onlinedocs/gcc/Invoking-gcov.html">gcov documentation</a>.
 */
public class GcovParser {

  private static final Logger logger = Logger.getLogger(GcovParser.class.getName());
  private List<SourceFileCoverage> allSourceFiles;
  private final InputStream inputStream;
  private SourceFileCoverage currentSourceFileCoverage;
  private ListMultimap<Integer, String> branchValues;

  private GcovParser(InputStream inputStream) {
    this.inputStream = inputStream;
  }

  public static List<SourceFileCoverage> parse(InputStream inputStream) throws IOException {
    return new GcovParser(inputStream).parse();
  }

  private List<SourceFileCoverage> parse() throws IOException {
    allSourceFiles = new ArrayList<>();
    boolean malformedInput = false;
    try (BufferedReader bufferedReader =
        new BufferedReader(new InputStreamReader(inputStream, UTF_8))) {
      String line;
      // TODO(bazel-team): This is susceptible to OOM if the input file is too large and doesn't
      // contain any newlines.
      while ((line = bufferedReader.readLine()) != null) {
        if (!parseLine(line)) {
          malformedInput = true;
        }
      }
    }
    endSourceFile();
    if (malformedInput) {
      logger.log(
          Level.WARNING,
          "gcov intermediate input was malformed, some lines might not have been parsed. "
              + "Check the previous log entries for more information.");
    }
    return allSourceFiles;
  }

  /**
   * Merges {@code currentSourceFileCoverage} into {@code allSourceFilesCoverageData} and resets
   * {@code currentSourceFileCoverage} to null.
   */
  private void endSourceFile() {
    if (currentSourceFileCoverage == null) {
      return;
    }
    recordBranchInformation(branchValues);
    allSourceFiles.add(currentSourceFileCoverage);
    currentSourceFileCoverage = null;
  }

  private boolean parseLine(String line) {
    if (line.isEmpty()) {
      return true;
    }
    if (line.startsWith(GCOV_FILE_MARKER)) {
      endSourceFile();
      return parseSource(line);
    }
    if (line.startsWith(GCOV_FUNCTION_MARKER)) {
      return parseFunction(line);
    }
    if (line.startsWith(GCOV_LINE_MARKER)) {
      return parseLCount(line);
    }
    if (line.startsWith(GCOV_BRANCH_MARKER)) {
      return parseBranch(line);
    }
    if (line.startsWith(GCOV_VERSION_MARKER) || line.startsWith(GCOV_CWD_MARKER)) {
      // Ignore these fields for now as they are not necessary.
      return true;
    }
    logger.log(
        Level.WARNING,
        "Line <" + line + "> does not respect the gcov intermediate format and was ignored.");
    return false;
  }

  private boolean parseSource(String line) {
    String sourcefile = line.substring(GCOV_FILE_MARKER.length());
    if (sourcefile.isEmpty()) {
      logger.log(Level.WARNING, "gcov info doesn't contain source file name on line: " + line);
      return false;
    }
    currentSourceFileCoverage = new SourceFileCoverage(sourcefile);
    branchValues = MultimapBuilder.treeKeys().arrayListValues().build();
    return true;
  }

  /**
   * Valid lines: function:start_line_number,end_line_number,execution_count,function_name
   * function:start_line_number,execution_count,function_name
   */
  private boolean parseFunction(String line) {
    String lineContent = line.substring(GCOV_FUNCTION_MARKER.length());
    String[] items = lineContent.split(DELIMITER, -1);
    if (items.length != 4 && items.length != 3) {
      logger.log(Level.WARNING, "gcov info contains invalid line " + line);
      return false;
    }
    try {
      // Ignore end_line_number since it's redundant information.
      int startLine = Integer.parseInt(items[0]);
      long execCount = items.length == 4 ? Long.parseLong(items[2]) : Long.parseLong(items[1]);
      String functionName = items.length == 4 ? items[3] : items[2];
      currentSourceFileCoverage.addLineNumber(functionName, startLine);
      currentSourceFileCoverage.addFunctionExecution(functionName, execCount);
    } catch (NumberFormatException e) {
      logger.log(Level.WARNING, "gcov info contains invalid line " + line);
      return false;
    }
    return true;
  }

  /**
   * Valid lines: lcount:line number,execution_count,has_unexecuted_block lcount:line
   * number,execution_count
   */
  private boolean parseLCount(String line) {
    String lineContent = line.substring(GCOV_LINE_MARKER.length());
    String[] items = lineContent.split(DELIMITER, -1);
    if (items.length != 3 && items.length != 2) {
      logger.log(Level.WARNING, "gcov info contains invalid line " + line);
      return false;
    }
    try {
      // Ignore has_unexecuted_block since it's not used.
      int lineNr = Integer.parseInt(items[0]);
      long execCount = Long.parseLong(items[1]);
      currentSourceFileCoverage.addLine(lineNr, execCount);
    } catch (NumberFormatException e) {
      logger.log(Level.WARNING, "gcov info contains invalid line " + line);
      return false;
    }
    return true;
  }

  /** Valid lines: branch:line number,taken string */
  private boolean parseBranch(String line) {
    // We can't add this to the source file object because we need to construct branch numbers,
    // which can only be done once we have all the branches for a given line number.
    String lineContent = line.substring(GCOV_BRANCH_MARKER.length());
    String[] items = lineContent.split(DELIMITER, -1);
    if (items.length != 2) {
      logger.log(Level.WARNING, "gcov info contains invalid line " + line);
      return false;
    }
    // Ignore has_unexecuted_block since it's not used.
    try {
      int lineNumber = Integer.parseInt(items[0]);
      String type = items[1];
      if (!(type.equals(GCOV_BRANCH_NOTEXEC)
          || type.equals(GCOV_BRANCH_NOTTAKEN)
          || type.equals(GCOV_BRANCH_TAKEN))) {
        logger.log(Level.WARNING, "gcov info contains invalid line " + line);
        return false;
      }
      branchValues.put(lineNumber, type);
    } catch (NumberFormatException e) {
      logger.log(Level.WARNING, "gcov info contains invalid line " + line);
      return false;
    }
    return true;
  }

  private void recordBranchInformation(ListMultimap<Integer, String> branchMap) {
    for (Map.Entry<Integer, Collection<String>> lineEntry : branchMap.asMap().entrySet()) {
      int branchNumber = 0;
      Collection<String> branches = lineEntry.getValue();
      for (String value : branches) {
        int execCount = 0;
        boolean evaluated = false;
        switch (value) {
          case GCOV_BRANCH_NOTEXEC:
            break;
          case GCOV_BRANCH_NOTTAKEN:
            evaluated = true;
            break;
          case GCOV_BRANCH_TAKEN:
            evaluated = true;
            execCount =
                1; // we don't have the number of executions recorded, so simply say "1" if the
            // branch was taken
            break;
          default:
            throw new AssertionError("Invalid branch value '" + value + "'");
        }
        currentSourceFileCoverage.addBranch(
            lineEntry.getKey(),
            BranchCoverage.createWithDummyBlock(
                lineEntry.getKey(), Integer.toString(branchNumber), evaluated, execCount));
        branchNumber++;
      }
    }
  }
}
