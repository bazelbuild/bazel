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
    Files.write(path, createScriptText(0, "Hello!", null));

    ProcessParameters parameters = createBuilder().build();
    ProcessResult result = new ProcessRunner(parameters, executorService).runSynchronously();

    Assert.assertEquals(0, result.exitCode());
    Assert.assertEquals("Hello!", result.outString());
    Assert.assertEquals("", result.errString());
  }

  @Test
  public void testFailure() throws Exception {
    Files.write(path, createScriptText(124, null, "Failure"));

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
    Files.write(path, Collections.singleton(isWindows() ? "set /p inp=type" : "read smthg"));

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
    Files.write(path, createScriptText(12, "Info\nMulti\nline", "Failure"));

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
      @Nullable final String output, @Nullable final String error) {
    List<String> text = Lists.newArrayList();
    if (isWindows()) {
      text.add("@echo off");
    }
    if (output != null) {
      text.add("echo \"" + output + "\"");
    }
    if (error != null) {
      text.add("echo \"" + error + "\" 1>&2");
    }
    text.add((isWindows() ? "exit /b " : "exit ") + exitCode);
    return text;
  }

  private static boolean isWindows() {
    return OS.WINDOWS.equals(OS.getCurrent());
  }
}
