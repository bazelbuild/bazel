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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.blackbox.framework.ProcessRunner.ProcessRunnerException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * Class for running Bazel process in the working directory of the blackbox test.
 *
 * <p>Provides customization methods for timeout, environment variables, etc., which modify the
 * current instance of <code>BuilderRunner</code>, and also return it for chain calls.
 *
 * <p>The instance keeps only parameters for the Bazel invocations, like timeout or environment
 * variables values, but not the data related to the actual Bazel invocations. That is why the same
 * instance can be used to invoke several commands.
 */
public final class BuilderRunner {
  private static final long DEFAULT_TIMEOUT_MILLIS = TimeUnit.SECONDS.toMillis(30);

  private final Path workDir;
  private final Path binaryPath;
  private final Map<String, String> env;
  private final ExecutorService executorService;
  private long timeoutMillis;
  private boolean useDefaultRc = true;
  private int errorCode = 0;
  private List<String> flags;
  private boolean shouldFail;

  /**
   * Creates the BuilderRunner
   *
   * @param workDir working directory of the test
   * @param binaryPath {@link Path} to the Bazel executable
   * @param defaultTimeoutMillis default timeout in milliseconds to use if the user has not
   *     specified timeout for Bazel command invocation
   * @param env environment variables to be passed to Bazel process (part, common for all Bazel
   *     invocations)
   * @param executorService {@link ExecutorService} to be used by the {@link ProcessRunner}, which
   *     actually invokes the Bzel command
   */
  BuilderRunner(
      Path workDir,
      Path binaryPath,
      long defaultTimeoutMillis,
      Map<String, String> env,
      ExecutorService executorService) {
    Preconditions.checkNotNull(workDir);
    Preconditions.checkNotNull(binaryPath);
    Preconditions.checkNotNull(env);
    Preconditions.checkState(defaultTimeoutMillis > 0, "Expected default timeout to be positive");

    this.workDir = workDir;
    this.binaryPath = binaryPath;
    this.env = Maps.newHashMap(env);
    this.executorService = executorService;
    this.timeoutMillis = defaultTimeoutMillis;
    this.flags = Lists.newArrayList();
  }

  /**
   * Sets environment variable for the Bazel invocation.
   *
   * @param name name of the variable
   * @param value value of variable
   * @return this BuildRunner instance
   */
  public BuilderRunner withEnv(String name, String value) {
    env.put(name, value);
    return this;
  }

  /**
   * Sets the expected error code.
   *
   * @param errorCode the value of the error code
   * @return this BuildRunner instance
   */
  public BuilderRunner withErrorCode(int errorCode) {
    this.errorCode = errorCode;
    return this;
  }

  /**
   * Expect Bazel to fail. This method is needed when the exact error code can not be specified.
   *
   * @return this BuildRunner instance
   */
  public BuilderRunner shouldFail() {
    this.shouldFail = true;
    return this;
  }

  /**
   * Sets timeout value for the Bazel process invocation. If not called, default value is used,
   * which is calculated from the test parameters. See {@link
   * BlackBoxTestContext#getTestTimeoutMillis()}. If the invocation time exceeds timeout, {@link
   * TimeoutException} is thrown.
   *
   * @param timeoutMillis timeout value in milliseconds
   * @return this BuilderRunner instance
   */
  public BuilderRunner withTimeout(long timeoutMillis) {
    Preconditions.checkState(this.timeoutMillis > 0);
    this.timeoutMillis = timeoutMillis;
    return this;
  }

  /**
   * Specifies that Bazel should not pass the default .bazelrc (in the test working directory) as
   * parameter
   *
   * @return this BuilderRunner instance
   */
  public BuilderRunner withoutDefaultRc() {
    useDefaultRc = false;
    return this;
  }

  /**
   * Specifies the flags to pass to Bazel.
   *
   * <p>We need it as a builder method, so that several consequent Bazel calls with the same set of
   * flags could be performed: bazel build --flag1 --flag2 //... bazel info --flag1 --flag2
   * bazel-bin
   *
   * @return this BuilderRunner instance
   */
  public BuilderRunner withFlags(String... flags) {
    Collections.addAll(this.flags, flags);
    return this;
  }

  /**
   * Runs <code>bazel info &lt;parameter&gt;</code> and returns the result. Asserts the process exit
   * code. Does not assert that the error stream is empty.
   *
   * @param parameters - info command parameter (can be omitted) and the Bazel flags. If Bazel was
   *     invoked with some flags, the same set of flags should be used with info.
   * @return ProcessResult with process exit code, strings with stdout and error streams contents
   * @throws TimeoutException in case of timeout
   * @throws IOException in case of the process startup/interaction problems
   * @throws InterruptedException if the current thread is interrupted while waiting
   * @throws ProcessRunnerException if the process return code is not zero or error stream is not
   *     empty when it was expected
   */
  public ProcessResult info(String... parameters) throws Exception {
    // additional expectations for the info time to be under the default timeout
    withTimeout(Math.min(DEFAULT_TIMEOUT_MILLIS, timeoutMillis));
    return runBinary("info", parameters);
  }

  /**
   * Runs <code>bazel help</code> and returns the result. Asserts the process exit code. Does not
   * assert that the error stream is empty.
   *
   * @return ProcessResult with process exit code, strings with stdout and error streams contents
   * @throws TimeoutException in case of timeout
   * @throws IOException in case of the process startup/interaction problems
   * @throws InterruptedException if the current thread is interrupted while waiting
   * @throws ProcessRunnerException if the process return code is not zero or error stream is not
   *     empty when it was expected
   */
  public ProcessResult help() throws Exception {
    // additional expectations for the info time to be under the default timeout
    withTimeout(Math.min(DEFAULT_TIMEOUT_MILLIS, timeoutMillis));
    return runBinary("help");
  }

  /**
   * Runs <code>bazel test &lt;args&gt;</code> and returns the result. Asserts that the process exit
   * code is zero. Does not assert that the error stream is empty.
   *
   * @param args arguments to pass to test command
   * @return ProcessResult with process exit code, strings with stdout and error streams contents
   * @throws TimeoutException in case of timeout
   * @throws IOException in case of the process startup/interaction problems
   * @throws InterruptedException if the current thread is interrupted while waiting
   * @throws ProcessRunnerException if the process return code is not zero or error stream is not
   *     empty when it was expected
   */
  public ProcessResult test(String... args) throws Exception {
    return runBinary("test", args);
  }

  /**
   * Runs <code>bazel build &lt;args&gt;</code> and returns the result. Asserts that the process
   * exit code is zero. Does not assert that the error stream is empty.
   *
   * @param args arguments to pass to build command
   * @return ProcessResult with process exit code, strings with stdout and error streams contents
   * @throws TimeoutException in case of timeout
   * @throws IOException in case of the process startup/interaction problems
   * @throws InterruptedException if the current thread is interrupted while waiting
   * @throws ProcessRunnerException if the process return code is not zero or error stream is not
   *     empty when it was expected
   */
  public ProcessResult build(String... args) throws Exception {
    return runBinary("build", args);
  }

  /**
   * Runs <code>bazel run &lt;args&gt;</code> and returns the result. Asserts that the process exit
   * code is zero. Does not assert that the error stream is empty.
   *
   * @param args arguments to pass to run command
   * @return ProcessResult with process exit code, strings with stdout and error streams contents
   * @throws TimeoutException in case of timeout
   * @throws IOException in case of the process startup/interaction problems
   * @throws InterruptedException if the current thread is interrupted while waiting
   * @throws ProcessRunnerException if the process return code is not zero or error stream is not
   *     empty when it was expected
   */
  public ProcessResult run(String... args) throws Exception {
    return runBinary("run", args);
  }

  /**
   * Runs <code>bazel shutdown</code> and returns the result. Asserts that the process exit code is
   * zero. Does not assert that the error stream is empty.
   *
   * @return ProcessResult with process exit code, strings with stdout and error streams contents
   * @throws TimeoutException in case of timeout
   * @throws IOException in case of the process startup/interaction problems
   * @throws InterruptedException if the current thread is interrupted while waiting
   * @throws ProcessRunnerException if the process return code is not zero or error stream is not
   *     empty when it was expected
   */
  public ProcessResult shutdown() throws Exception {
    // additional expectations for the shutdown time to be under the default timeout
    withTimeout(Math.min(DEFAULT_TIMEOUT_MILLIS, timeoutMillis));
    return runBinary("shutdown");
  }

  private ProcessResult runBinary(String command, String... args) throws Exception {
    List<String> list = Lists.newArrayList();

    if (useDefaultRc) {
      Path bazelRc = workDir.resolve(".bazelrc");
      if (Files.exists(bazelRc)) {
        list.add("--bazelrc");
        list.add(bazelRc.toAbsolutePath().toString());
      }
    }
    list.add(command);
    list.addAll(this.flags);
    Collections.addAll(list, args);

    ProcessParameters parameters =
        ProcessParameters.builder()
            .setWorkingDirectory(workDir.toFile())
            .setName(binaryPath.toString())
            .setTimeoutMillis(timeoutMillis)
            .setArguments(list)
            .setEnvironment(ImmutableMap.copyOf(env))
            // bazel writes info messages to error stream, so
            // we need to allow the error output stream be not empty
            .setExpectedEmptyError(false)
            .setExpectedExitCode(errorCode)
            .setExpectedToFail(shouldFail)
            .build();
    return new ProcessRunner(parameters, executorService).runSynchronously();
  }
}
