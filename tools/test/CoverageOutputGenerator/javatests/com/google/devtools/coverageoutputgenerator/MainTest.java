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
import static java.util.stream.Collectors.joining;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Stream;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.rules.TestName;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link Main}. */
@RunWith(JUnit4.class)
public class MainTest {

  private static final int TEST_PARSE_PARALLELISM = 4;
  @Rule public TemporaryFolder temporaryFolder = new TemporaryFolder();
  @Rule public TestName testName = new TestName();
  private Path coverageDir;

  @Before
  public void createCoverageDirectory() throws IOException {
    coverageDir = temporaryFolder.newFolder("coverage-dir").toPath();
  }

  @Test
  public void testMainEmptyCoverageDir() {
    assertThat(Main.getCoverageFilesInDir(coverageDir)).isEmpty();
  }

  @Test
  public void testMainGetCoverageFilesInDir() throws IOException {
    Path ccCoverageDir = Files.createTempDirectory(coverageDir, "cc_coverage");
    Path javaCoverageDir = Files.createTempDirectory(coverageDir, "java_coverage");

    Files.createTempFile(ccCoverageDir, "tracefile1", ".dat");
    Files.createTempFile(javaCoverageDir, "tracefile2", ".dat");

    List<Path> coverageFiles = Main.getCoverageFilesInDir(coverageDir);
    assertThat(coverageFiles).hasSize(2);
  }

  @Test
  public void testParallelParse_1KLoC_1KLcovFiles() throws Exception {
    assertParallelParse(1024, 4, 256);
  }

  @Test
  public void testParallelParse_1MLoC_4LcovFiles() throws Exception {
    assertParallelParse(4, 1024, 1024);
  }

  @Test
  public void testParallelParse_1MLoC_1LcovFiles() throws Exception {
    assertParallelParse(1, 1024, 1024);
  }

  @Test
  public void testEmptyInputProducesEmptyOutput() throws Exception {
    Path output =
        Paths.get(
            temporaryFolder.getRoot().getAbsolutePath(),
            testName.getMethodName() + ".coverage.dat");
    int exitCode =
        Main.runWithArgs(
            "--coverage_dir", coverageDir.toAbsolutePath().toString(),
            "--output_file", output.toAbsolutePath().toString());
    assertThat(exitCode).isEqualTo(0);
    assertThat(output.toFile().exists()).isTrue();
    assertThat(output.toFile().length()).isEqualTo(0L);
  }

  @Test
  public void testNonEmptyInputProducesNonEmptyOutput() throws Exception {
    LcovMergerTestUtils.generateLcovFiles("test_data/simple_test", 8, 8, 8, coverageDir);
    Path output =
        Paths.get(
            temporaryFolder.getRoot().getAbsolutePath(),
            testName.getMethodName() + ".coverage.dat");
    int exitCode =
        Main.runWithArgs(
            "--coverage_dir", coverageDir.toAbsolutePath().toString(),
            "--output_file", output.toAbsolutePath().toString());
    assertThat(exitCode).isEqualTo(0);
    assertThat(output.toFile().exists()).isTrue();
    assertThat(output.toFile().length()).isGreaterThan(0L);
  }

  @Test
  public void testNestedCoverageFiles() throws Exception {
    var directFiles =
        LcovMergerTestUtils.generateLcovFiles("test_data/direct", 2, 2, 8, coverageDir);
    var nestedDir = coverageDir.resolve("nested");
    LcovMergerTestUtils.generateLcovFiles("test_data/nested", 2, 2, 8, nestedDir);

    Path reportsFile =
        Paths.get(
            temporaryFolder.getRoot().getAbsolutePath(), testName.getMethodName() + ".reports.txt");
    Files.writeString(
        reportsFile,
        Stream.concat(directFiles.stream(), Stream.of(nestedDir))
            .map(Path::toString)
            .collect(joining("\n")));

    Path output =
        Paths.get(
            temporaryFolder.getRoot().getAbsolutePath(),
            testName.getMethodName() + ".coverage.dat");
    int exitCode =
        Main.runWithArgs(
            "--reports_file", reportsFile.toString(),
            "--output_file", output.toAbsolutePath().toString());
    assertThat(exitCode).isEqualTo(0);
    assertThat(output.toFile().exists()).isTrue();
    var outputContent = Files.readString(output);
    assertThat(outputContent).contains("SF:test_data/direct0");
    assertThat(outputContent).contains("SF:test_data/direct1");
    assertThat(outputContent).contains("SF:test_data/nested0");
    assertThat(outputContent).contains("SF:test_data/nested1");
  }

  private void assertParallelParse(int numLcovFiles, int numSourceFiles, int numLinesPerSourceFile)
      throws Exception {

    ByteArrayOutputStream sequentialOutput = new ByteArrayOutputStream();
    ByteArrayOutputStream parallelOutput = new ByteArrayOutputStream();

    LcovMergerTestUtils.generateLcovFiles(
        "test_data/simple_test", numLcovFiles, numSourceFiles, numLinesPerSourceFile, coverageDir);

    List<Path> coverageFiles = Main.getCoverageFilesInDir(coverageDir);

    Coverage sequentialCoverage = Main.parseFilesSequentially(coverageFiles);
    LcovPrinter.print(sequentialOutput, sequentialCoverage);

    Coverage parallelCoverage = Main.parseFilesInParallel(coverageFiles, TEST_PARSE_PARALLELISM);
    LcovPrinter.print(parallelOutput, parallelCoverage);

    assertThat(parallelOutput.toString()).isEqualTo(sequentialOutput.toString());
  }
}
