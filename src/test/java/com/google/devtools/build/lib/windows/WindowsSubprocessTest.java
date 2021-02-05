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

package com.google.devtools.build.lib.windows;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.runfiles.Runfiles;
import java.io.File;
import java.io.InputStream;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link WindowsSubprocess}.
 */
@RunWith(JUnit4.class)
@TestSpec(localOnly = true, supportedOs = OS.WINDOWS)
public class WindowsSubprocessTest {
  private String mockSubprocess;
  private String mockBinary;
  private Subprocess process;
  private Runfiles runfiles;

  @Before
  public void loadJni() throws Exception {
    runfiles = Runfiles.create();
    mockSubprocess =
        runfiles.rlocation(
            "io_bazel/src/test/java/com/google/devtools/build/lib/windows/MockSubprocess_deploy.jar");
    mockBinary = System.getProperty("java.home") + "\\bin\\java.exe";

    process = null;
  }

  @After
  public void terminateProcess() throws Exception {
    if (process != null) {
      process.destroy();
      process.close();
      process = null;
    }
  }

  @Test
  public void testSystemRootIsSetByDefault() throws Exception {
    SubprocessBuilder subprocessBuilder = new SubprocessBuilder(WindowsSubprocessFactory.INSTANCE);
    subprocessBuilder.setWorkingDirectory(new File("."));
    subprocessBuilder.setArgv(ImmutableList.of(mockBinary, "-jar", mockSubprocess, "O$SYSTEMROOT"));
    process = subprocessBuilder.start();
    process.waitFor();
    assertThat(process.exitValue()).isEqualTo(0);

    byte[] buf = new byte[11];
    process.getInputStream().read(buf);
    assertThat(new String(buf, UTF_8).trim()).isEqualTo(System.getenv("SYSTEMROOT").trim());
  }

  @Test
  public void testSystemDriveIsSetByDefault() throws Exception {
    SubprocessBuilder subprocessBuilder = new SubprocessBuilder(WindowsSubprocessFactory.INSTANCE);
    subprocessBuilder.setWorkingDirectory(new File("."));
    subprocessBuilder.setArgv(
        ImmutableList.of(mockBinary, "-jar", mockSubprocess, "O$SYSTEMDRIVE"));
    process = subprocessBuilder.start();
    process.waitFor();
    assertThat(process.exitValue()).isEqualTo(0);

    byte[] buf = new byte[3];
    process.getInputStream().read(buf);
    assertThat(new String(buf, UTF_8).trim()).isEqualTo(System.getenv("SYSTEMDRIVE").trim());
  }

  @Test
  public void testSystemRootIsSet() throws Exception {
    SubprocessBuilder subprocessBuilder = new SubprocessBuilder(WindowsSubprocessFactory.INSTANCE);
    subprocessBuilder.setWorkingDirectory(new File("."));
    subprocessBuilder.setArgv(ImmutableList.of(mockBinary, "-jar", mockSubprocess, "O$SYSTEMROOT"));
    // Case shouldn't matter on Windows
    subprocessBuilder.setEnv(ImmutableMap.of("SystemRoot", "C:\\MySystemRoot"));
    process = subprocessBuilder.start();
    process.waitFor();
    assertThat(process.exitValue()).isEqualTo(0);

    byte[] buf = new byte[16];
    process.getInputStream().read(buf);
    assertThat(new String(buf, UTF_8).trim()).isEqualTo("C:\\MySystemRoot");
  }

  @Test
  public void testSystemDriveIsSet() throws Exception {
    SubprocessBuilder subprocessBuilder = new SubprocessBuilder(WindowsSubprocessFactory.INSTANCE);
    subprocessBuilder.setWorkingDirectory(new File("."));
    subprocessBuilder.setArgv(
        ImmutableList.of(mockBinary, "-jar", mockSubprocess, "O$SYSTEMDRIVE"));
    // Case shouldn't matter on Windows
    subprocessBuilder.setEnv(ImmutableMap.of("SystemDrive", "X:"));
    process = subprocessBuilder.start();
    process.waitFor();
    assertThat(process.exitValue()).isEqualTo(0);

    byte[] buf = new byte[3];
    process.getInputStream().read(buf);
    assertThat(new String(buf, UTF_8).trim()).isEqualTo("X:");
  }

  @Test
  public void testStreamAvailable_zeroAfterClose() throws Exception {
    SubprocessBuilder subprocessBuilder = new SubprocessBuilder(WindowsSubprocessFactory.INSTANCE);
    subprocessBuilder.setWorkingDirectory(new File("."));
    subprocessBuilder.setArgv(ImmutableList.of(mockBinary, "-jar", mockSubprocess, "OHELLO"));
    process = subprocessBuilder.start();
    InputStream inputStream = process.getInputStream();
    // We don't know if the process has already written to the pipe
    assertThat(inputStream.available()).isAnyOf(0, 5);
    process.waitFor();
    // Windows allows streams to be read after the process has died.
    assertThat(inputStream.available()).isAnyOf(0, 5);
    inputStream.close();
    assertThat(assertThrows(IllegalStateException.class, inputStream::available))
        .hasMessageThat()
        .contains("Stream already closed");
  }

  /**
   * An argument and its command-line-escaped counterpart.
   *
   * <p>Such escaping ensures that Bazel correctly forwards arguments to subprocesses.
   */
  private static final class ArgPair {
    public final String original;
    public final String escaped;

    public ArgPair(String original, String escaped) {
      this.original = original;
      this.escaped = escaped;
    }
  };

  /** Asserts that a subprocess correctly receives command line arguments. */
  private void assertSubprocessReceivesArgsAsIntended(ArgPair... args) throws Exception {
    // Look up the path of the printarg.exe utility.
    String printArgExe =
        runfiles.rlocation(
            "io_bazel/src/test/java/com/google/devtools/build/lib/windows/printarg.exe");
    assertThat(printArgExe).isNotEmpty();

    for (ArgPair arg : args) {
      // Assert that the command-line encoding logic works as intended.
      assertThat(ShellUtils.windowsEscapeArg(arg.original)).isEqualTo(arg.escaped);

      // Create a separate subprocess just for this argument.
      SubprocessBuilder subprocessBuilder =
          new SubprocessBuilder(WindowsSubprocessFactory.INSTANCE);
      subprocessBuilder.setWorkingDirectory(new File("."));
      subprocessBuilder.setArgv(ImmutableList.of(printArgExe, arg.original));
      process = subprocessBuilder.start();
      process.waitFor();
      assertThat(process.exitValue()).isEqualTo(0);

      // The subprocess printed its argv[1] in parentheses, e.g. (foo).
      // Assert that it printed exactly the *original* argument in parentheses.
      byte[] buf = new byte[1000];
      process.getInputStream().read(buf);
      String actual = new String(buf, UTF_8).trim();
      assertThat(actual).isEqualTo("(" + arg.original + ")");
    }
  }

  @Test
  public void testSubprocessReceivesArgsAsIntended() throws Exception {
    assertSubprocessReceivesArgsAsIntended(
        new ArgPair("", "\"\""),
        new ArgPair(" ", "\" \""),
        new ArgPair("\"", "\"\\\"\""),
        new ArgPair("\"\\", "\"\\\"\\\\\""),
        new ArgPair("\\", "\\"),
        new ArgPair("\\\"", "\"\\\\\\\"\""),
        new ArgPair("with space", "\"with space\""),
        new ArgPair("with^caret", "with^caret"),
        new ArgPair("space ^caret", "\"space ^caret\""),
        new ArgPair("caret^ space", "\"caret^ space\""),
        new ArgPair("with\"quote", "\"with\\\"quote\""),
        new ArgPair("with\\backslash", "with\\backslash"),
        new ArgPair("one\\ backslash and \\space", "\"one\\ backslash and \\space\""),
        new ArgPair("two\\\\backslashes", "two\\\\backslashes"),
        new ArgPair("two\\\\ backslashes \\\\and space", "\"two\\\\ backslashes \\\\and space\""),
        new ArgPair("one\\\"x", "\"one\\\\\\\"x\""),
        new ArgPair("two\\\\\"x", "\"two\\\\\\\\\\\"x\""),
        new ArgPair("a \\ b", "\"a \\ b\""),
        new ArgPair("a \\\" b", "\"a \\\\\\\" b\""),
        new ArgPair("A", "A"),
        new ArgPair("\"a\"", "\"\\\"a\\\"\""),
        new ArgPair("B C", "\"B C\""),
        new ArgPair("\"b c\"", "\"\\\"b c\\\"\""),
        new ArgPair("D\"E", "\"D\\\"E\""),
        new ArgPair("\"d\"e\"", "\"\\\"d\\\"e\\\"\""),
        new ArgPair("C:\\F G", "\"C:\\F G\""),
        new ArgPair("\"C:\\f g\"", "\"\\\"C:\\f g\\\"\""),
        new ArgPair("C:\\H\"I", "\"C:\\H\\\"I\""),
        new ArgPair("\"C:\\h\"i\"", "\"\\\"C:\\h\\\"i\\\"\""),
        new ArgPair("C:\\J\\\"K", "\"C:\\J\\\\\\\"K\""),
        new ArgPair("\"C:\\j\\\"k\"", "\"\\\"C:\\j\\\\\\\"k\\\"\""),
        new ArgPair("C:\\L M ", "\"C:\\L M \""),
        new ArgPair("\"C:\\l m \"", "\"\\\"C:\\l m \\\"\""),
        new ArgPair("C:\\N O\\", "\"C:\\N O\\\\\""),
        new ArgPair("\"C:\\n o\\\"", "\"\\\"C:\\n o\\\\\\\"\""),
        new ArgPair("C:\\P Q\\ ", "\"C:\\P Q\\ \""),
        new ArgPair("\"C:\\p q\\ \"", "\"\\\"C:\\p q\\ \\\"\""),
        new ArgPair("C:\\R\\S\\", "C:\\R\\S\\"),
        new ArgPair("C:\\R x\\S\\", "\"C:\\R x\\S\\\\\""),
        new ArgPair("\"C:\\r\\s\\\"", "\"\\\"C:\\r\\s\\\\\\\"\""),
        new ArgPair("\"C:\\r x\\s\\\"", "\"\\\"C:\\r x\\s\\\\\\\"\""),
        new ArgPair("C:\\T U\\W\\", "\"C:\\T U\\W\\\\\""),
        new ArgPair("\"C:\\t u\\w\\\"", "\"\\\"C:\\t u\\w\\\\\\\"\""));
  }
}
