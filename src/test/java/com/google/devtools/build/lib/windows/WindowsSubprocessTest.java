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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.runfiles.Runfiles;
import java.io.File;
import java.nio.charset.Charset;
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
  private static final Charset UTF8 = Charset.forName("UTF-8");
  private String mockSubprocess;
  private String mockBinary;
  private Subprocess process;

  @Before
  public void loadJni() throws Exception {
    Runfiles runfiles = Runfiles.create();
    mockSubprocess =
        runfiles.rlocation(
            "io_bazel/src/test/java/com/google/devtools/build/lib/MockSubprocess_deploy.jar");
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
    SubprocessBuilder subprocessBuilder = new SubprocessBuilder();
    subprocessBuilder.setWorkingDirectory(new File("."));
    subprocessBuilder.setSubprocessFactory(WindowsSubprocessFactory.INSTANCE);
    subprocessBuilder.setArgv(mockBinary, "-jar", mockSubprocess, "O$SYSTEMROOT");
    process = subprocessBuilder.start();
    process.waitFor();
    assertThat(process.exitValue()).isEqualTo(0);

    byte[] buf = new byte[11];
    process.getInputStream().read(buf);
    assertThat(new String(buf, UTF8).trim()).isEqualTo(System.getenv("SYSTEMROOT").trim());
  }

  @Test
  public void testSystemDriveIsSetByDefault() throws Exception {
    SubprocessBuilder subprocessBuilder = new SubprocessBuilder();
    subprocessBuilder.setWorkingDirectory(new File("."));
    subprocessBuilder.setSubprocessFactory(WindowsSubprocessFactory.INSTANCE);
    subprocessBuilder.setArgv(mockBinary, "-jar", mockSubprocess, "O$SYSTEMDRIVE");
    process = subprocessBuilder.start();
    process.waitFor();
    assertThat(process.exitValue()).isEqualTo(0);

    byte[] buf = new byte[3];
    process.getInputStream().read(buf);
    assertThat(new String(buf, UTF8).trim()).isEqualTo(System.getenv("SYSTEMDRIVE").trim());
  }

  @Test
  public void testSystemRootIsSet() throws Exception {
    SubprocessBuilder subprocessBuilder = new SubprocessBuilder();
    subprocessBuilder.setWorkingDirectory(new File("."));
    subprocessBuilder.setSubprocessFactory(WindowsSubprocessFactory.INSTANCE);
    subprocessBuilder.setArgv(mockBinary, "-jar", mockSubprocess, "O$SYSTEMROOT");
    // Case shouldn't matter on Windows
    subprocessBuilder.setEnv(ImmutableMap.of("SystemRoot", "C:\\MySystemRoot"));
    process = subprocessBuilder.start();
    process.waitFor();
    assertThat(process.exitValue()).isEqualTo(0);

    byte[] buf = new byte[16];
    process.getInputStream().read(buf);
    assertThat(new String(buf, UTF8).trim()).isEqualTo("C:\\MySystemRoot");
  }

  @Test
  public void testSystemDriveIsSet() throws Exception {
    SubprocessBuilder subprocessBuilder = new SubprocessBuilder();
    subprocessBuilder.setWorkingDirectory(new File("."));
    subprocessBuilder.setSubprocessFactory(WindowsSubprocessFactory.INSTANCE);
    subprocessBuilder.setArgv(mockBinary, "-jar", mockSubprocess, "O$SYSTEMDRIVE");
    // Case shouldn't matter on Windows
    subprocessBuilder.setEnv(ImmutableMap.of("SystemDrive", "X:"));
    process = subprocessBuilder.start();
    process.waitFor();
    assertThat(process.exitValue()).isEqualTo(0);

    byte[] buf = new byte[3];
    process.getInputStream().read(buf);
    assertThat(new String(buf, UTF8).trim()).isEqualTo("X:");
  }
}
