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

package com.google.devtools.build.lib.windows.jni;

import java.util.List;

/** Process management on Windows. */
public class WindowsProcesses {
  public static final long INVALID = -1;

  private WindowsProcesses() {
    // Prevent construction
  }

  /**
   * Creates a process with the specified Windows command line.
   *
   * <p>Appropriately quoting arguments is the responsibility of the caller.
   *
   * @param argv0 the binary to run; must be unquoted; must be either an absolute, normalized
   *     Windows path with a drive letter (e.g. "c:\foo\bar app.exe") or a single file name (e.g.
   *     "foo app.exe")
   * @param argvRest the rest of the command line, i.e. argv[1:] (needs to be quoted Windows style)
   * @param env the environment of the new process. null means inherit that of the Bazel server
   * @param cwd the working directory of the new process. if null, the same as that of the current
   *     process
   * @param stdoutFile the file the stdout should be redirected to. if null, nativeReadStdout will
   *     work.
   * @param stderrFile the file the stdout should be redirected to. if null, nativeReadStderr will
   *     work.
   * @param redirectErrorStream whether we merge the process's standard error and standard output.
   * @return the opaque identifier of the created process
   */
  public static long createProcess(
      String argv0,
      String argvRest,
      byte[] env,
      String cwd,
      String stdoutFile,
      String stderrFile,
      boolean redirectErrorStream) {
    WindowsJniLoader.loadJni();
    return nativeCreateProcess(
        argv0, argvRest, env, cwd, stdoutFile, stderrFile, redirectErrorStream);
  }

  public static long createProcess(
      String argv0, String argvRest, byte[] env, String cwd, String stdoutFile, String stderrFile) {
    WindowsJniLoader.loadJni();
    return nativeCreateProcess(argv0, argvRest, env, cwd, stdoutFile, stderrFile, false);
  }

  private static native long nativeCreateProcess(
      String argv0,
      String argvRest,
      byte[] env,
      String cwd,
      String stdoutFile,
      String stderrFile,
      boolean redirectErrorStream);

  /**
   * Writes data from the given array to the stdin of the specified process.
   *
   * <p>Blocks until either some data was written or the process is terminated.
   *
   * @return the number of bytes written
   */
  public static int writeStdin(long process, byte[] bytes, int offset, int length) {
    WindowsJniLoader.loadJni();
    return nativeWriteStdin(process, bytes, offset, length);
  }

  private static native int nativeWriteStdin(long process, byte[] bytes, int offset, int length);

  /** Returns an opaque identifier of stdout stream for the process. */
  public static long getStdout(long process) {
    WindowsJniLoader.loadJni();
    return nativeGetStdout(process);
  }

  private static native long nativeGetStdout(long process);

  /** Returns an opaque identifier of stderr stream for the process. */
  public static long getStderr(long process) {
    WindowsJniLoader.loadJni();
    return nativeGetStderr(process);
  }

  private static native long nativeGetStderr(long process);

  /**
   * Reads data from the stream into the given array. {@code stream} should come from {@link
   * #nativeGetStdout(long)} or {@link #nativeGetStderr(long)}.
   *
   * <p>Blocks until either some data was read or the process is terminated.
   *
   * @return the number of bytes read, 0 on EOF, or -1 if there was an error.
   */
  public static int readStream(long stream, byte[] bytes, int offset, int length) {
    WindowsJniLoader.loadJni();
    return nativeReadStream(stream, bytes, offset, length);
  }

  private static native int nativeReadStream(long stream, byte[] bytes, int offset, int length);

  /**
   * Waits until the given process terminates. If timeout is non-negative, it indicates the number
   * of milliseconds before the call times out.
   *
   * <p>Return values:
   * <li>0: Process finished
   * <li>1: Timeout
   * <li>2: Something went wrong
   */
  public static int waitFor(long process, long timeout) {
    WindowsJniLoader.loadJni();
    return nativeWaitFor(process, timeout);
  }

  private static native int nativeWaitFor(long process, long timeout);

  /**
   * Returns the exit code of the process. Throws {@code IllegalStateException} if something goes
   * wrong.
   */
  public static int getExitCode(long process) {
    WindowsJniLoader.loadJni();
    return nativeGetExitCode(process);
  }

  private static native int nativeGetExitCode(long process);

  /** Returns the process ID of the given process or -1 if there was an error. */
  public static int getProcessPid(long process) {
    WindowsJniLoader.loadJni();
    return nativeGetProcessPid(process);
  }

  private static native int nativeGetProcessPid(long process);

  /** Terminates the given process. Returns true if the termination was successful. */
  public static boolean terminate(long process) {
    WindowsJniLoader.loadJni();
    return nativeTerminate(process);
  }

  private static native boolean nativeTerminate(long process);

  /**
   * Releases the native data structures associated with the process.
   *
   * <p>Calling any other method on the same process after this call will result in the JVM crashing
   * or worse.
   */
  public static void deleteProcess(long process) {
    WindowsJniLoader.loadJni();
    nativeDeleteProcess(process);
  }

  private static native void nativeDeleteProcess(long process);

  /**
   * Closes the stream
   *
   * @param stream should come from {@link #nativeGetStdout(long)} or {@link
   *     #nativeGetStderr(long)}.
   */
  public static void closeStream(long stream) {
    WindowsJniLoader.loadJni();
    nativeCloseStream(stream);
  }

  private static native void nativeCloseStream(long stream);

  /**
   * Returns a string representation of the last error caused by any call on the given process or
   * the empty string if the last operation was successful.
   *
   * <p>Does <b>NOT</b> terminate the process if it is still running.
   *
   * <p>After this call returns, subsequent calls will return the empty string if there was no
   * failed operation in between.
   */
  public static String processGetLastError(long process) {
    WindowsJniLoader.loadJni();
    return nativeProcessGetLastError(process);
  }

  private static native String nativeProcessGetLastError(long process);

  public static String streamGetLastError(long process) {
    WindowsJniLoader.loadJni();
    return nativeStreamGetLastError(process);
  }

  private static native String nativeStreamGetLastError(long process);

  /** returns the PID of the current process. */
  public static int getpid() {
    WindowsJniLoader.loadJni();
    return nativeGetpid();
  }

  private static native int nativeGetpid();

  // TODO(laszlocsomor): Replace this method with ShellUtils.windowsEscapeArg in order to fix
  // https://github.com/bazelbuild/bazel/issues/7122
  public static String quoteCommandLine(List<String> argv) {
    StringBuilder result = new StringBuilder();
    for (int iArg = 0; iArg < argv.size(); iArg++) {
      if (iArg != 0) {
        result.append(" ");
      }
      String arg = argv.get(iArg);
      if (arg.isEmpty()) {
        result.append("\"\"");
        continue;
      }
      boolean hasSpace = arg.contains(" ");
      if (!arg.contains("\"") && !arg.contains("\\") && !hasSpace) {
        // fast path. Just append the input string.
        result.append(arg);
      } else {
        // We need to quote things if the argument contains a space.
        if (hasSpace) {
          result.append("\"");
        }

        for (int iChar = 0; iChar < arg.length(); iChar++) {
          boolean lastChar = iChar == arg.length() - 1;
          switch (arg.charAt(iChar)) {
            case '"':
              // Escape double quotes
              result.append("\\\"");
              break;
            case '\\':
              // Backslashes at the end of the string are quoted if we add quotes
              if (lastChar) {
                result.append(hasSpace ? "\\\\" : "\\");
              } else {
                // Backslashes everywhere else are quoted if they are followed by a
                // quote or a backslash
                result.append(
                    arg.charAt(iChar + 1) == '"' || arg.charAt(iChar + 1) == '\\' ? "\\\\" : "\\");
              }
              break;
            default:
              result.append(arg.charAt(iChar));
          }
        }
        // Add ending quotes if we added a quote at the beginning.
        if (hasSpace) {
          result.append("\"");
        }
      }
    }

    return result.toString();
  }
}
