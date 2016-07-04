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

import com.google.common.base.Charsets;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.shell.SubprocessBuilder.StreamAction;

import java.io.File;
import java.io.IOException;
import java.util.Map;

/**
 * A subprocess factory that uses the Win32 API.
 */
public class WindowsSubprocessFactory implements Subprocess.Factory {
  public static final WindowsSubprocessFactory INSTANCE = new WindowsSubprocessFactory();

  private WindowsSubprocessFactory() {
    // Singleton
  }

  @Override
  public Subprocess create(SubprocessBuilder builder) throws IOException {
    WindowsJniLoader.loadJni();
    
    String commandLine = WindowsProcesses.quoteCommandLine(builder.getArgv());
    byte[] env = builder.getEnv() == null ? null : convertEnvToNative(builder.getEnv());

    String stdoutPath = getRedirectPath(builder.getStdout(), builder.getStdoutFile());
    String stderrPath = getRedirectPath(builder.getStderr(), builder.getStderrFile());

    long nativeProcess = WindowsProcesses.nativeCreateProcess(
        commandLine, env, builder.getWorkingDirectory().getPath(), stdoutPath, stderrPath);
    String error = WindowsProcesses.nativeGetLastError(nativeProcess);
    if (!error.isEmpty()) {
      WindowsProcesses.nativeDelete(nativeProcess);
      throw new IOException(error);
    }

    return new WindowsSubprocess(nativeProcess, stdoutPath != null, stderrPath != null);
  }

  private String getRedirectPath(StreamAction action, File file) {
    switch (action) {
      case DISCARD:
        return "NUL";  // That's /dev/null on Windows

      case REDIRECT:
        return file.getPath();

      case STREAM:
        return null;

      default:
        throw new IllegalStateException();
    }
  }

  /**
   * Converts an environment map to the format expected in lpEnvironment by CreateProcess().
   */
  private byte[] convertEnvToNative(Map<String, String> env) throws IOException {
    if (env.isEmpty()) {
      // Special case: CreateProcess() always expects the environment block to be terminated
      // with two zeros.
      return new byte[] { 0, 0, };
    }

    StringBuilder result = new StringBuilder();

    for (String key : Ordering.natural().sortedCopy(env.keySet())) {
      if (key.contains("=")) {
        throw new IOException("Environment variable names must not contain '='");
      }
      result.append(key + "=" + env.get(key) + "\0");
    }

    result.append("\0");
    return result.toString().getBytes(Charsets.UTF_8);
  }
}
