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

import java.io.File;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A component that converts language specific raw coverage data to pseudo lcov format.
 */
class LcovMerger {
  private static final Logger logger = Logger.getLogger(LcovMerger.class.getName());

  private final String outputFile;
  private final List<File> fileList;

  /**
   * Constructs an {@link LcovMerger} and collects the raw coverage files.
   */
  LcovMerger(String originalCoverageFilesDirectory, String generatedCoverageDataOutputPath) {
    this.fileList = getDatFiles(originalCoverageFilesDirectory);
    this.outputFile = generatedCoverageDataOutputPath;
  }

  /**
   * Merge all files in {@link fileList} and write to {@link outputFile}.
   *
   * @return successful or not
   */
  boolean merge() {
    if (fileList.isEmpty()) {
      logger.log(Level.SEVERE, "No lcov file found.");
      return false;
    }
    if (fileList.size() > 1) {
      logger.log(Level.SEVERE, "Only one lcov file supported now, but found " + fileList.size());
      return false;
    }
    try {
      Files.copy(
          fileList.get(0).toPath(), Paths.get(outputFile), StandardCopyOption.REPLACE_EXISTING);
    } catch (IOException e) {
      logger.log(Level.SEVERE, "Failed to copy file: " + e.getMessage());
      return false;
    }
    return true;
  }

  private List<File> getDatFiles(String coverageDir) {
    List<File> datFiles = new ArrayList<>();
    try (DirectoryStream<Path> stream = Files.newDirectoryStream(Paths.get(coverageDir), "*.dat")) {
      for (Path entry : stream) {
        datFiles.add(entry.toFile());
      }
    } catch (IOException x) {
      logger.log(Level.SEVERE, "error reading folder " + coverageDir + ": " + x.getMessage());
    }
    return datFiles;
  }
}
