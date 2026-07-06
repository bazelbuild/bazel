// Copyright 2016 The Bazel Authors. All Rights Reserved.
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
package com.google.testing.coverage;


import com.google.common.collect.ImmutableSet;
import com.google.testing.coverage.CoverageData.BranchData;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Map;
import java.util.Optional;

/**
 * Simple lcov formatter to be used with lcov_merger.par.
 *
 * <p>The lcov format is documented here: http://ltp.sourceforge.net/coverage/lcov/geninfo.1.php
 */
public class JacocoLCOVFormatter {

  // The delimiter between the provided "exec path" and the source file name if an explicit mapping
  // is provided via an "execPath//sourcePath" line in the paths-for-coverage.txt file.
  private static final String EXEC_PATH_DELIMITER = "///";

  // The "exec paths" of files that may have coverage data. A source file may have a computed path
  // of "com/google/.../Foo.java" (corresponding to the class path) but the actual path is likely to
  // be "java/com/google/.../Foo.java" or similar. That is, it will have an additional prefix.
  // We use this set to remap the source file path to the actual path. If it is not set, we assume
  // provided source paths are correct.
  private final Optional<ImmutableSet<String>> sourceExecPaths;

  public JacocoLCOVFormatter(ImmutableSet<String> sourceExecPaths) {
    this.sourceExecPaths = Optional.of(sourceExecPaths);
  }

  public JacocoLCOVFormatter() {
    this.sourceExecPaths = Optional.empty();
  }

  /**
   * Writes the given coverage data to the given writer in LCOV format.
   *
   * @param writer the writer to write the coverage data to
   * @param coverageData a map of source file name to the coverage data for that file.
   * @throws IOException if an error occurs while writing the coverage data
   */
  public void writeCoverageData(PrintWriter writer, Map<String, CoverageData> coverageData)
      throws IOException {
    for (Map.Entry<String, CoverageData> entry : coverageData.entrySet()) {
      Optional<String> execPath = getExecPath(entry.getKey());
      if (!execPath.isPresent()) {
        continue;
      }
      CoverageData coverage = entry.getValue();
      writeSourceFile(writer, execPath.get(), coverage);
    }
    writer.flush();
  }

  private Optional<String> getExecPath(String sourceFile) {
    if (!sourceExecPaths.isPresent()) {
      return Optional.of(sourceFile);
    }

    String matchingFileName = sourceFile.startsWith("/") ? sourceFile : "/" + sourceFile;
    for (String execPath : sourceExecPaths.get()) {
      if (execPath.contains(EXEC_PATH_DELIMITER)) {
        String[] parts = execPath.split(EXEC_PATH_DELIMITER, 2);
        if (parts.length != 2) {
          continue;
        }
        if (parts[1].equals(matchingFileName)) {
          return Optional.of(parts[0]);
        }
      } else if (execPath.endsWith(matchingFileName) || execPath.equals(sourceFile)) {
        return Optional.of(execPath);
      }
    }
    return Optional.empty();
  }

  private void writeSourceFile(PrintWriter writer, String sourceFile, CoverageData coverage) {
    writer.printf("SF:%s\n", sourceFile);

    for (String methodName : coverage.getMethods()) {
      int line = coverage.getMethodLine(methodName);
      boolean executed = coverage.isMethodExecuted(methodName);
      writer.printf("FN:%d,%s\n", line, methodName);
      writer.printf("FNDA:%d,%s\n", executed ? 1 : 0, methodName);
    }

    for (Integer line : coverage.linesWithBranches()) {
      BranchData branchData = coverage.getBranches(line);
      int numBranches = branchData.size();
      boolean executed = branchData.anyBranchTaken();
      if (executed) {
        for (int branchIdx = 0; branchIdx < numBranches; branchIdx++) {
          // We haven't got execution counts for branches; just record if they were hit or not.
          if (branchData.isBranchTaken(branchIdx)) {
            writer.printf("BRDA:%d,%d,%d,%d\n", line, 0, branchIdx, 1); // executed, taken
          } else {
            writer.printf("BRDA:%d,%d,%d,%d\n", line, 0, branchIdx, 0); // executed, not taken
          }
        }
      } else {
        for (int branchIdx = 0; branchIdx < numBranches; branchIdx++) {
          writer.printf("BRDA:%d,%d,%d,%s\n", line, 0, branchIdx, "-"); // not executed
        }
      }
    }
    for (int line : coverage.getInstrumentedLines()) {
      writer.printf("DA:%d,%d\n", line, coverage.isLineExecuted(line) ? 1 : 0);
    }
    writer.println("end_of_record");
  }
}
