// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.blackbox.framework;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.Maps;
import com.google.devtools.build.lib.util.OS;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Holds JUnit integration tests context, passed to tests from base class.
 *
 * <p>Provides access to source and work directories, generated and binary files, allows to run
 * built binary, creates {@link BuilderRunner} for running Bazel commands
 */
public final class BlackBoxTestContext {
  private final Path workDir;
  private final Path srcDir;
  private final Path tmpDir;
  private final Map<String, String> commonEnv;
  private final ExecutorService executorService;
  private final String productName;
  private final Path binaryPath;
  private Path genFilesPath;
  private Path binFilesPaths;

  public BlackBoxTestContext(
      String testName,
      String productName,
      Path binaryPath,
      Map<String, String> commonEnv,
      ExecutorService executorService)
      throws Exception {
    assertThat(Files.exists(binaryPath)).isTrue();

    this.commonEnv = Maps.newHashMap(commonEnv);
    this.productName = productName;
    this.binaryPath = binaryPath;
    srcDir = getPathFromEnv("TEST_SRCDIR");
    tmpDir = getPathFromEnv("TEST_TMPDIR");

    workDir = tmpDir.resolve(testName);
    Files.createDirectories(workDir);

    this.executorService = executorService;
  }

  /** Returns the product name: bazel */
  public String getProductName() {
    return productName;
  }

  /** Returns the working directory of the test */
  public Path getWorkDir() {
    return workDir;
  }

  /** Returns the source directory (TEST_SRCDIR) of the test. */
  public Path getSrcDir() {
    return srcDir;
  }

  /** Returns the temp directory (TEST_TMPDIR) of the test */
  public Path getTmpDir() {
    return tmpDir;
  }

  /**
   * Writes <code>lines</code> using ISO_8859_1 into the file, specified by the <code>subPath</code>
   * relative to the working directory. Overrides the file if it exists, creates the file if it does
   * not exist.
   *
   * @param subPath path to file relative to working directory
   * @param lines lines of text to write. Newlines are added by the method.
   * @return Path to the file
   * @throws IOException in case if the file can not be created/overridden, or can not be open for
   *     writing
   */
  public Path write(String subPath, String... lines) throws IOException {
    return PathUtils.writeFileInDir(workDir, subPath, lines);
  }

  /**
   * Writes <code>lines</code> using ISO_8859_1 into the BUILD file in the directory, specified by
   * the <code>subPath</code> relative to the working directory. Overrides the file if it exists,
   * creates the file if it does not exist.
   *
   * @param subPathToDir path to directory relative to working directory, where BUILD file should be
   *     written
   * @param lines lines of text to write. Newlines are added by the method.
   * @return Path to the file
   * @throws IOException in case if the file can not be created/overridden, or can not be open for
   *     writing
   */
  public Path writeBuild(String subPathToDir, String... lines) throws IOException {
    String separator = (subPathToDir.endsWith(File.separator) ? "" : File.separator);
    return write(subPathToDir + separator + "BUILD", lines);
  }

  /**
   * Reads the lines of the file, specified by the <code>subPath</code> relative to the working
   * directory, in ISO_8859_1.
   *
   * @param subPath path to the file relative to the working directory
   * @return list of file lines (without the newline characters)
   * @throws IOException if the file does not exist or can not be read
   */
  public List<String> read(String subPath) throws IOException {
    return PathUtils.readFile(workDir, subPath);
  }

  /**
   * Resolve a path relative to "bazel-genfiles".
   *
   * <p>Calls <code>bazel info bazel-genfiles</code>, caches the result.
   *
   * @param bazel the instance of BuilderRunner to run info with
   * @param subPathUnderGen path to the file under bazel-gen directory
   * @return full path to the resolved file
   * @throws Exception if <code>bazel info</code> command fails
   */
  public Path resolveGenPath(BuilderRunner bazel, String subPathUnderGen) throws Exception {
    if (genFilesPath == null) {
      genFilesPath = PathUtils.resolve(workDir, bazel.info(productName + "-genfiles").outString());
    }
    return PathUtils.resolve(genFilesPath, subPathUnderGen);
  }

  /**
   * Resolve a path relative to "bazel-bin".
   *
   * <p>Calls <code>bazel info bazel-bin</code>
   *
   * @param bazel the instance of BuilderRunner to run info with
   * @param subPathUnderBin path to the file under bazel-bin directory
   * @return full path to the resolved file
   * @throws Exception if <code>bazel info</code> command fails
   */
  public Path resolveBinPath(BuilderRunner bazel, String subPathUnderBin) throws Exception {
    Path binPath = PathUtils.resolve(workDir, bazel.info(productName + "-bin").outString());
    return PathUtils.resolve(binPath, subPathUnderBin);
  }

  /**
   * Resolve a path relative to "execution_root".
   * Useful for checking the contents of the generated external repositories.
   *
   * <p>Calls <code>bazel info execution_root</code>
   *
   * @param bazel the instance of BuilderRunner to run info with
   * @param subPathUnderBin path to the file under execution_root directory
   * @return full path to the resolved file
   * @throws Exception if <code>bazel info</code> command fails
   */
  public Path resolveExecRootPath(BuilderRunner bazel, String subPathUnderBin) throws Exception {
    Path binPath = PathUtils.resolve(workDir, bazel.info("execution_root").outString());
    return PathUtils.resolve(binPath, subPathUnderBin);
  }

  /**
   * Runs the built executable. Calls <code>bazel info</code> to get the information about bazel-bin
   * directory location.
   *
   * @param bazel the instance of BuilderRunner to run info with
   * @param subPathUnderBin path to the executable relative to bazel-bin directory
   * @param timeoutMillis timeout on the process execution
   * @return ProcessResult result of the execution with the process exit code and strings with
   *     stdout and stderr contents.
   * @throws Exception if <code>bazel info</code> command fails or executable invocation fails.
   */
  public ProcessResult runBuiltBinary(
      BuilderRunner bazel, String subPathUnderBin, long timeoutMillis) throws Exception {
    if (OS.WINDOWS.equals(OS.getCurrent()) && !subPathUnderBin.endsWith(".exe")) {
      subPathUnderBin += ".exe";
    }
    Path executable = resolveBinPath(bazel, subPathUnderBin);
    assertThat(Files.exists(executable)).isTrue();
    assertThat(Files.isRegularFile(executable)).isTrue();
    assertThat(Files.isExecutable(executable)).isTrue();

    ProcessParameters parameters =
        ProcessParameters.builder()
            .setWorkingDirectory(workDir.toFile())
            .setName(executable.toString())
            .setTimeoutMillis(getProcessTimeoutMillis(timeoutMillis))
            .build();
    return new ProcessRunner(parameters, executorService).runSynchronously();
  }

  /**
   * Creates the instance of BuilderRunner for running Bazel commands in the working directory. see
   * {@link BuilderRunner}
   *
   * @return BuilderRunner interface for running Bazel commands.
   */
  public BuilderRunner bazel() {
    return new BuilderRunner(
        workDir, binaryPath, getProcessTimeoutMillis(-1), commonEnv, executorService);
  }

  /**
   * Take the value from environment variable and assert that it is a path, and the file or
   * directory, specified by this path, exists.
   *
   * @param name name of the environment variable
   * @return Path to the file where the value of environment variable points
   */
  private static Path getPathFromEnv(String name) {
    String pathStr = System.getenv(name);
    assertThat(pathStr).isNotNull();
    Path path = Paths.get(pathStr);
    assertThat(Files.exists(path)).isTrue();
    return path.toAbsolutePath();
  }

  /**
   * Define the value of the timeout for the Bazel process invoked for the test. Use the value,
   * specified by the user, or default test timeout value.
   *
   * @param timeoutMillis value for the timeout, specified by the user. If the user has not
   *     specified the value, -1 is passed.
   * @return timeout value in milliseconds
   */
  private static long getProcessTimeoutMillis(long timeoutMillis) {
    if (timeoutMillis > 0) {
      return timeoutMillis;
    }
    return getTestTimeoutMillis();
  }

  /**
   * Determine the timeout of the blackbox test, use information from the environment variable.
   *
   * @return timeout value in milliseconds
   */
  private static long getTestTimeoutMillis() {
    String timeout = System.getenv("TEST_TIMEOUT");
    if (timeout != null) {
      try {
        return TimeUnit.SECONDS.toMillis(Integer.parseInt(timeout));
      } catch (NumberFormatException e) {
        System.out.println("Invalid test timeout value, using default.");
      }
    }
    return TimeUnit.SECONDS.toMillis(900);
  }
}
