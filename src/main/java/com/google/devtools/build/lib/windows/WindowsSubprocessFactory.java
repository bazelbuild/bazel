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
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.shell.SubprocessBuilder.StreamAction;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

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

    List<String> argv = builder.getArgv();

    // DO NOT quote argv0, nativeCreateProcess will do it for us.
    String argv0 = processArgv0(argv.get(0));
    String argvRest =
        argv.size() > 1 ? WindowsProcesses.quoteCommandLine(argv.subList(1, argv.size())) : "";
    byte[] env = builder.getEnv() == null ? null : convertEnvToNative(builder.getEnv());

    String stdoutPath = getRedirectPath(builder.getStdout(), builder.getStdoutFile());
    String stderrPath = getRedirectPath(builder.getStderr(), builder.getStderrFile());

    long nativeProcess =
        WindowsProcesses.nativeCreateProcess(
            argv0, argvRest, env, builder.getWorkingDirectory().getPath(), stdoutPath, stderrPath);
    String error = WindowsProcesses.nativeProcessGetLastError(nativeProcess);
    if (!error.isEmpty()) {
      WindowsProcesses.nativeDeleteProcess(nativeProcess);
      throw new IOException(error);
    }

    return new WindowsSubprocess(
        nativeProcess,
        argv0 + " " + argvRest,
        stdoutPath != null,
        stderrPath != null,
        builder.getTimeoutMillis());
  }

  public String processArgv0(String argv0) {
    // Normalize the path and make it Windows-style.
    // If argv0 is at least MAX_PATH (260 chars) long, createNativeProcess calls GetShortPathNameW
    // to obtain a 8dot3 name for it (thereby support long paths in CreateProcessA), but then argv0
    // must be prefixed with "\\?\" for GetShortPathNameW to work, so it also must be an absolute,
    // normalized, Windows-style path.
    // Therefore if it's absolute, then normalize it also.
    // If it's not absolute, then it cannot be longer than MAX_PATH, since MAX_PATH also limits the
    // length of file names.
    PathFragment argv0fragment = PathFragment.create(argv0);
    return (argv0fragment.isAbsolute())
        ? argv0fragment.normalize().getPathString().replace('/', '\\')
        : argv0;
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

  private String getSystemRoot(Map<String, String> env) {
    // Windows environment variables are case-insensitive, so we can't just say
    // System.getenv().get("SystemRoot")
    for (String key : env.keySet()) {
      if (key.toUpperCase().equals("SYSTEMROOT")) {
        return env.get(key);
      }
    }

    return null;
  }

  /**
   * Converts an environment map to the format expected in lpEnvironment by CreateProcess().
   */
  private byte[] convertEnvToNative(Map<String, String> env) throws IOException {
    Map<String, String> realEnv = new TreeMap<>();
    realEnv.putAll(env == null ? System.getenv() : env);
    if (getSystemRoot(realEnv) == null) {
      // Some versions of MSVCRT.DLL require SystemRoot to be set. It's quite a common library to
      // link in, so we add this environment variable regardless of whether the caller requested
      // it or not.
      String systemRoot = getSystemRoot(System.getenv());
      if (systemRoot != null) {
        realEnv.put("SystemRoot", systemRoot);
      }
    }

    if (realEnv.isEmpty()) {
      // Special case: CreateProcess() always expects the environment block to be terminated
      // with two zeros.
      return new byte[] { 0, 0, };
    }

    StringBuilder result = new StringBuilder();
    for (Map.Entry<String, String> entry : realEnv.entrySet()) {
      if (entry.getKey().contains("=")) {
        // lpEnvironment requires no '=' in environment variable name, but on Windows,
        // System.getenv() returns environment variables like '=C:' or '=ExitCode', so it can't
        // be an error, we have ignore them here.
        continue;
      }
      result.append(entry.getKey() + "=" + entry.getValue() + "\0");
    }

    result.append("\0");
    return result.toString().getBytes(Charsets.UTF_8);
  }
}
