/*
 * // Copyright 2018 The Bazel Authors. All rights reserved.
 * //
 * // Licensed under the Apache License, Version 2.0 (the "License");
 * // you may not use this file except in compliance with the License.
 * // You may obtain a copy of the License at
 * //
 * // http://www.apache.org/licenses/LICENSE-2.0
 * //
 * // Unless required by applicable law or agreed to in writing, software
 * // distributed under the License is distributed on an "AS IS" BASIS,
 * // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * // See the License for the specific language governing permissions and
 * // limitations under the License.
 */

package com.google.devtools.build.lib.integration.blackbox.framework;

import com.google.common.collect.Lists;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.integration.blackbox.framework.ProcessParameters.Builder;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.OsUtils;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ProcessRunnerTest {
  private static ExecutorService executorService;
  private Path directory;
  private Path path;

  @BeforeClass
  public static void setUpExecutor() {
    executorService = MoreExecutors.getExitingExecutorService(
            (ThreadPoolExecutor) Executors.newFixedThreadPool(2), 1, TimeUnit.SECONDS);
  }

  @Before
  public void setUp() throws Exception {
    directory = Files.createTempDirectory(getClass().getSimpleName());
    path = Files.createTempFile(directory, "script", isWindows() ? ".bat" : "");
    Assert.assertTrue(Files.exists(path));
    Assert.assertTrue(path.toFile().setExecutable(true));
    path.toFile().deleteOnExit();
    directory.toFile().deleteOnExit();
  }

  @AfterClass
  public static void tearDownExecutor() {
    MoreExecutors.shutdownAndAwaitTermination(executorService, 5, TimeUnit.SECONDS);
  }

  @Test
  public void testSuccess() throws Exception {
    Files.write(path, createScriptText(0, Collections.singletonList("Hello!"), null));

    ProcessParameters parameters = createBuilder().build();
    ProcessResult result = new ProcessRunner(parameters, executorService).runSynchronously();

    Assert.assertEquals(0, result.exitCode());
    Assert.assertEquals("Hello!", result.outString());
    Assert.assertEquals("", result.errString());
  }

  @Test
  public void testFailure() throws Exception {
    Files.write(path, createScriptText(124, null, Collections.singletonList("Failure")));

    ProcessParameters parameters = createBuilder()
        .setExpectedExitCode(124)
        .setExpectedEmptyError(false)
        .build();
    ProcessResult result = new ProcessRunner(parameters, executorService).runSynchronously();

    Assert.assertEquals(124, result.exitCode());
    Assert.assertEquals("", result.outString());
    Assert.assertEquals("Failure", result.errString());
  }

  @Test
  public void testTimeout() throws Exception {
    Files.write(path, Collections.singleton(isWindows()
        ? "%systemroot%\\system32\\cmd.exe /C \"start /I /B powershell -Version 3.0 -NoLogo -Sta" +
        " -NoProfile -InputFormat Text -OutputFormat Text -NonInteractive" +
        " -Command \"\"&PowerShell Sleep 10\""
        : "read smthg"));

    ProcessParameters parameters = createBuilder()
        .setExpectedExitCode(-1)
        .setExpectedEmptyError(false)
        .setTimeoutMillis(100)
        .build();
    try {
      new ProcessRunner(parameters, executorService).runSynchronously();
      Assert.assertTrue(false);
    } catch (TimeoutException e) {
      // ignore
    }
  }

  @Test
  public void testRedirect() throws Exception {
    Files.write(path, createScriptText(12, Lists.newArrayList("Info", "Multi", "line"),
        Collections.singletonList("Failure")));

    Path out = directory.resolve("out.txt");
    Path err = directory.resolve("err.txt");

    try {
      ProcessParameters parameters = createBuilder()
          .setExpectedExitCode(12)
          .setExpectedEmptyError(false)
          .setRedirectOutput(out)
          .setRedirectError(err)
          .build();
      ProcessResult result = new ProcessRunner(parameters, executorService).runSynchronously();

      Assert.assertEquals(12, result.exitCode());
      Assert.assertEquals("Info\nMulti\nline", result.outString());
      Assert.assertEquals("Failure", result.errString());
    } finally {
      Files.delete(out);
      Files.delete(err);
    }
  }

  private Builder createBuilder() {
    return ProcessParameters.builder()
        .setWorkingDirectory(directory.toFile())
        .setName(path.toAbsolutePath().toString());
  }

  private static List<String> createScriptText(final int exitCode,
      @Nullable final List<String> output, @Nullable final List<String> error) {
    List<String> text = Lists.newArrayList();
    if (isWindows()) {
      text.add("@echo off");
    }
    text.addAll(echoStrings(output, ""));
    text.addAll(echoStrings(error, isWindows() ? ">&2" : " 1>&2"));
    text.add((isWindows() ? "exit /b " : "exit ") + exitCode);
    return text;
  }

  private static List<String> echoStrings(@Nullable List<String> input, String redirect) {
    if (input == null) return Collections.emptyList();
    String quote = isWindows() ? "" : "\"";
    return input.stream()
        .map(s -> String.format("echo %s%s%s%s", quote, s, quote, redirect))
        .collect(Collectors.toList());
  }

  private static boolean isWindows() {
    return OS.WINDOWS.equals(OS.getCurrent());
  }
}
