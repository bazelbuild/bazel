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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Helper class for creating and parsing lcov tracefiles and the necessary data structured used by
 * {@code LcovMerger}.
 */
public final class LcovMergerTestUtils {

  private LcovMergerTestUtils() {}

  static List<String> generateLcovContents(
      String srcPrefix, int numSourceFiles, int numLinesPerSourceFile) {
    ArrayList<String> lines = new ArrayList<>();
    for (int i = 0; i < numSourceFiles; i++) {
      lines.add(String.format("SF:%s%s.cc", srcPrefix, i));
      lines.add("FNF:0");
      lines.add("FNH:0");
      for (int srcLineNum = 1; srcLineNum <= numLinesPerSourceFile; srcLineNum += 4) {
        lines.add(String.format("BA:%s,2", srcLineNum));
      }
      lines.add("BRF:" + numLinesPerSourceFile / 4);
      lines.add("BRH:" + numLinesPerSourceFile / 4);
      for (int srcLineNum = 1; srcLineNum <= numLinesPerSourceFile; srcLineNum++) {
        lines.add(String.format("DA:%s,%s", srcLineNum, srcLineNum % 2));
      }
      lines.add("LH:" + numLinesPerSourceFile / 2);
      lines.add("LF:" + numLinesPerSourceFile);
      lines.add("end_of_record");
    }
    return lines;
  }

  static List<Path> generateLcovFiles(
      String srcPrefix, int numLcovFiles, int numSrcFiles, int numLinesPerSrcFile, Path coverageDir)
      throws IOException {
    Files.createDirectories(coverageDir);
    Path lcovFile = Files.createFile(Paths.get(coverageDir.toString(), "coverage0.dat"));
    List<Path> lcovFiles = new ArrayList<>();
    Files.write(lcovFile, generateLcovContents(srcPrefix, numSrcFiles, numLinesPerSrcFile));
    lcovFiles.add(lcovFile);
    for (int i = 1; i < numLcovFiles; i++) {
      lcovFiles.add(
          Files.createSymbolicLink(
              Paths.get(coverageDir.toString(), String.format("coverage%s.dat", i)), lcovFile));
    }
    return lcovFiles;
  }
}
