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

package com.google.devtools.build.lib.windows.util;

import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.fail;

import com.google.common.base.Joiner;
import com.google.devtools.build.lib.windows.WindowsJniLoader;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/** Utilities for running Java tests on Windows. */
public final class WindowsTestUtil {
  private WindowsTestUtil() {}

  private static Map<String, String> runfiles;

  public static void loadJni() throws Exception {
    String jniDllPath = WindowsTestUtil.getRunfile("io_bazel/src/main/native/windows_jni.dll");
    WindowsJniLoader.loadJniForTesting(jniDllPath);
  }

  // Do not use WindowsFileSystem.createDirectoryJunction but reimplement junction creation here.
  // If that method were buggy, using it here would compromise the test.
  public static void createJunctions(String scratchRoot, Map<String, String> links)
      throws Exception {
    List<String> args = new ArrayList<>();
    boolean first = true;

    // Shell out to cmd.exe to create all junctions in one go.
    // Running "cmd.exe /c command1 arg1 arg2 && command2 arg1 ... argN && ..." will run all
    // commands within one cmd.exe invocation.
    for (Map.Entry<String, String> e : links.entrySet()) {
      if (first) {
        args.add("cmd.exe /c");
        first = false;
      } else {
        args.add("&&");
      }

      args.add(
          String.format(
              "mklink /j \"%s/%s\" \"%s/%s\"", scratchRoot, e.getKey(), scratchRoot, e.getValue()));
    }
    runCommand(args);
  }

  public static void deleteAllUnder(String path) throws IOException {
    if (new File(path).exists()) {
      runCommand("cmd.exe /c rd /s /q \"" + path + "\"");
    }
  }

  private static void runCommand(List<String> args) throws IOException {
    runCommand(Joiner.on(' ').join(args));
  }

  private static void runCommand(String cmd) throws IOException {
    Process p = Runtime.getRuntime().exec(cmd);
    try {
      // Wait no more than 5 seconds to create all junctions.
      p.waitFor(5, TimeUnit.SECONDS);
    } catch (InterruptedException e) {
      fail("Failed to execute command; cmd: " + cmd);
    }
    assertWithMessage("Command failed: " + cmd).that(p.exitValue()).isEqualTo(0);
  }

  public static String getRunfile(String runfilesPath) throws IOException {
    ensureRunfilesParsed();
    return runfiles.get(runfilesPath);
  }

  private static synchronized void ensureRunfilesParsed() throws IOException {
    if (runfiles != null) {
      return;
    }

    runfiles = new HashMap<>();
    InputStream fis = new FileInputStream(System.getenv("RUNFILES_MANIFEST_FILE"));
    InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
    try (BufferedReader br = new BufferedReader(isr)) {
      String line;
      while ((line = br.readLine()) != null) {
        String[] splitLine = line.split(" "); // This is buggy when the path contains spaces
        if (splitLine.length != 2) {
          continue;
        }

        runfiles.put(splitLine[0], splitLine[1]);
      }
    }
  }
}
