// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.gson.Gson;
import com.google.gson.annotations.SerializedName;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.zip.GZIPInputStream;

/**
 * A {@link Parser} for gcov intermediate json format introduced in GCC 9.1. See the flag {@code
 * --intermediate-format} in <a
 * href="https://gcc.gnu.org/onlinedocs/gcc-9.3.0/gcc/Invoking-Gcov.html">gcov documentation</a>.
 */
public class GcovJsonParser {
  private static final Logger logger = Logger.getLogger(GcovJsonParser.class.getName());
  private final InputStream inputStream;

  private GcovJsonParser(InputStream inputStream) {
    this.inputStream = inputStream;
  }

  public static List<SourceFileCoverage> parse(InputStream inputStream) throws IOException {
    return new GcovJsonParser(inputStream).parse();
  }

  private List<SourceFileCoverage> parse() throws IOException {
    ArrayList<SourceFileCoverage> allSourceFiles = new ArrayList<>();
    try (InputStream gzipStream = new GZIPInputStream(inputStream)) {
      ByteArrayOutputStream contents = new ByteArrayOutputStream();
      byte[] buffer = new byte[1024];
      int length;
      while ((length = gzipStream.read(buffer)) != -1) {
        contents.write(buffer, 0, length);
      }
      Gson gson = new Gson();
      GcovJsonFormat document = gson.fromJson(contents.toString(), GcovJsonFormat.class);
      if (!document.format_version.equals("1")) {
        logger.log(
            Level.WARNING,
            "Expect GCov JSON format version 1, got format version " + document.format_version);
      }
      for (GcovJsonFile file : document.files) {
        SourceFileCoverage currentFileCoverage = new SourceFileCoverage(file.file);
        for (GcovJsonFunction function : file.functions) {
          currentFileCoverage.addLineNumber(function.name, function.start_line);
          currentFileCoverage.addFunctionExecution(function.name, function.execution_count);
        }
        for (GcovJsonLine line : file.lines) {
          currentFileCoverage.addLine(
              line.line_number, LineCoverage.create(line.line_number, line.count, null));
          int branchNumber = 0;
          boolean taken = Arrays.stream(line.branches).anyMatch(b -> b.count > 0);
          for (GcovJsonBranch branch : line.branches) {
            currentFileCoverage.addBranch(
                line.line_number,
                BranchCoverage.createWithDummyBlock(
                    line.line_number, Integer.toString(branchNumber), taken, branch.count));
            branchNumber += 1;
          }
        }
        allSourceFiles.add(currentFileCoverage);
      }
    }

    return allSourceFiles;
  }

  // Classes for the Gson data mapper representing the structure of the GCov JSON format
  // These do not follow the Java naming styleguide as they need to match the JSON field names
  // Documentation can be found in GCov's manpage, of which the source is available at
  // https://gcc.gnu.org/git/?p=gcc.git;a=blob;f=gcc/doc/gcov.texi;h=dcdd7831ff063483d43e5347af0b67083c85ecc4;hb=4212a6a3e44f870412d9025eeb323fd4f50a61da#l184

  static class GcovJsonFormat {
    String gcc_version;
    GcovJsonFile[] files;
    String format_version;
    String current_working_directory;
    String data_file;
  }

  static class GcovJsonFile {
    String file;
    GcovJsonFunction[] functions;
    GcovJsonLine[] lines;
  }

  static class GcovJsonFunction {
    int blocks;
    int end_column;
    int start_line;
    String name;
    int blocks_executed;
    long execution_count;
    String demangled_name;
    int start_column;
    int end_line;
  }

  static class GcovJsonLine {
    GcovJsonBranch[] branches;
    long count;
    int line_number;
    boolean unexecuted_block;
    String function_name;
  }

  static class GcovJsonBranch {
    boolean fallthrough;
    long count;

    @SerializedName("throw")
    boolean _throw;
  }
}
