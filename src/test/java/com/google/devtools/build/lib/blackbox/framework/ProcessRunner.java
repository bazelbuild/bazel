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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.io.LineReader;
import com.google.devtools.build.lib.util.StringUtilities;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * Helper class for running Bazel process as external process from JUnit tests. Can be used to run
 * arbitrary external process and explore the results.
 */
public final class ProcessRunner {
  private static final Logger logger = Logger.getLogger(ProcessRunner.class.getName());
  private final ProcessParameters parameters;
  private final ExecutorService executorService;

  /**
   * Creates ProcessRunner
   *
   * @param parameters process parameters like executable name, arguments, timeout etc
   * @param executorService to use for process output/error streams reading; intentionally passed as
   *     a parameter so we can use the thread pool to speed up. Should be multi-threaded, as two
   *     separate tasks are submitted, to read from output and error streams.
   *     <p>SuppressWarnings: WeakerAccess - suppress the warning about constructor being public:
   *     the class is intended to be used outside the package. (IDE currently marks the possibility
   *     for the constructor to be package-private because the current usages are only inside the
   *     package, but it is going to change)
   */
  @SuppressWarnings("WeakerAccess")
  public ProcessRunner(ProcessParameters parameters, ExecutorService executorService) {
    this.parameters = parameters;
    this.executorService = executorService;
  }

  public ProcessResult runSynchronously() throws Exception {
    ImmutableList<String> args = parameters.arguments();
    final List<String> commandParts = new ArrayList<>(args.size() + 1);
    commandParts.add(parameters.name());
    commandParts.addAll(args);

    logger.info("Running: " + commandParts.stream().collect(Collectors.joining(" ")));

    ProcessBuilder processBuilder = new ProcessBuilder(commandParts);
    processBuilder.directory(parameters.workingDirectory());
    parameters.environment().ifPresent(map -> processBuilder.environment().putAll(map));

    parameters.redirectOutput().ifPresent(path -> processBuilder.redirectOutput(path.toFile()));
    parameters.redirectError().ifPresent(path -> processBuilder.redirectError(path.toFile()));

    Process process = processBuilder.start();

    try (ProcessStreamReader outReader =
        parameters.redirectOutput().isPresent()
            ? null
            : createReader(process.getInputStream(), ">> ");
        ProcessStreamReader errReader =
            parameters.redirectError().isPresent()
                ? null
                : createReader(process.getErrorStream(), "ERROR: ")) {

      long timeoutMillis = parameters.timeoutMillis();
      if (!process.waitFor(timeoutMillis, TimeUnit.MILLISECONDS)) {
        throw new TimeoutException(
            String.format(
                "%s timed out after %d seconds (%d millis)",
                parameters.name(), timeoutMillis / 1000, timeoutMillis));
      }

      List<String> err =
          errReader != null
              ? errReader.get()
              : Files.readAllLines(parameters.redirectError().get());
      List<String> out =
          outReader != null
              ? outReader.get()
              : Files.readAllLines(parameters.redirectOutput().get());

      int exitValue = process.exitValue();
      boolean expectedToFail = parameters.expectedToFail() || parameters.expectedExitCode() != 0;
      if ((exitValue == 0) == expectedToFail) {
        throw new ProcessRunnerException(
            String.format(
                "Expected to %s, but %s.\nError: %s\nOutput: %s",
                expectedToFail ? "fail" : "succeed",
                exitValue == 0 ? "succeeded" : "failed",
                StringUtilities.joinLines(err),
                StringUtilities.joinLines(out)));
      }
      // We want to check the exact exit code if it was explicitly set to something;
      // we already checked the variant when it is equal to zero above.
      if (parameters.expectedExitCode() != 0 && parameters.expectedExitCode() != exitValue) {
        throw new ProcessRunnerException(
            String.format(
                "Expected exit code %d, but found %d.\nError: %s\nOutput: %s",
                parameters.expectedExitCode(),
                exitValue,
                StringUtilities.joinLines(err),
                StringUtilities.joinLines(out)));
      }

      if (parameters.expectedEmptyError()) {
        if (!err.isEmpty()) {
          throw new ProcessRunnerException(
              "Expected empty error stream, but found: " + StringUtilities.joinLines(err));
        }
      }
      return ProcessResult.create(exitValue, out, err);
    } finally {
      process.destroy();
    }
  }

  private ProcessStreamReader createReader(InputStream stream, String prefix) {
    return new ProcessStreamReader(executorService, stream, s -> logger.fine(prefix + s));
  }

  /** Specific runtime exception for external process errors */
  public static class ProcessRunnerException extends RuntimeException {
    ProcessRunnerException(String message) {
      super(message);
    }
  }

  private static class ProcessStreamReader implements AutoCloseable {

    private final InputStream stream;
    private final Future<List<String>> future;
    private final AtomicReference<IOException> exception = new AtomicReference<>();

    private ProcessStreamReader(
        ExecutorService executorService,
        InputStream stream,
        @Nullable Consumer<String> logConsumer) {
      this.stream = stream;
      future =
          executorService.submit(
              () -> {
                final List<String> lines = Lists.newArrayList();
                try (BufferedReader reader =
                    new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
                  LineReader lineReader = new LineReader(reader);
                  String line;
                  while ((line = lineReader.readLine()) != null) {
                    if (logConsumer != null) {
                      logConsumer.accept(line);
                    }
                    lines.add(line);
                  }
                } catch (IOException e) {
                  exception.set(e);
                }
                return lines;
              });
    }

    public List<String> get()
        throws InterruptedException, ExecutionException, TimeoutException, IOException {
      try {
        List<String> lines = future.get(15, TimeUnit.SECONDS);
        if (exception.get() != null) {
          throw exception.get();
        }
        return lines;
      } finally {
        // if future is timed out
        stream.close();
      }
    }

    @Override
    public void close() throws Exception {
      stream.close();
    }
  }
}
