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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.coverageoutputgenerator.Constants.TRACEFILE_EXTENSION;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link Main}. */
@RunWith(JUnit4.class)
public class MainTest {

  private Path coverageDir;

  @Before
  public void createCoverageDirectory() throws IOException {
    coverageDir = Files.createTempDirectory("coverage-dir");
  }

  @Test
  public void testMainEmptyCoverageDir() {
    assertThat(Main.getCoverageFilesInDir(coverageDir.toAbsolutePath().toString())).isEmpty();
  }

  @Test
  public void testMainGetLcovTracefiles() throws IOException {
    Path ccCoverageDir = Files.createTempDirectory(coverageDir, "cc_coverage");
    Path javaCoverageDir = Files.createTempDirectory(coverageDir, "java_coverage");

    Files.createTempFile(ccCoverageDir, "tracefile1", ".dat");
    Files.createTempFile(javaCoverageDir, "tracefile2", ".dat");

    List<File> coverageFiles = Main.getCoverageFilesInDir(coverageDir.toAbsolutePath().toString());
    List<File> tracefiles = Main.getFilesWithExtension(coverageFiles, TRACEFILE_EXTENSION);
    assertThat(tracefiles).hasSize(2);
  }
}
