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

import com.google.devtools.build.lib.jni.JniLoader;

/** Process management on Windows. */
public class WindowsProcesses {
  public static final long INVALID = -1;

  static {
    JniLoader.loadJni();
  }

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
   * @param cwd the working directory of the new process. If null, the same as that of the current
   *     process.
   * @param stdoutFile the file the stdout should be redirected to. If null, {@link #getStdout} will
   *     work.
   * @param stderrFile the file the stdout should be redirected to. If null, {@link #getStderr} will
   *     work.
   * @param redirectErrorStream whether we merge the process's standard error and standard output.
   * @return the opaque identifier of the created process
   */
  public static native long createProcess(
      String argv0,
      String argvRest,
      byte[] env,
      String cwd,
      String stdoutFile,
      String stderrFile,
      boolean redirectErrorStream);

  public static long createProcess(
      String argv0, String argvRest, byte[] env, String cwd, String stdoutFile, String stderrFile) {
    return createProcess(argv0, argvRest, env, cwd, stdoutFile, stderrFile, false);
  }

  /**
   * Writes data from the given array to the stdin of the specified process.
   *
   * <p>Blocks until either some data was written or the process is terminated.
   *
   * @return the number of bytes written, or -1 if an error occurs.
   */
  public static native int writeStdin(long process, byte[] bytes, int offset, int length);

  /** Closes the stdin of the specified process. */
  public static native void closeStdin(long process);

  /** Returns an opaque identifier of stdout stream for the process. */
  public static native long getStdout(long process);

  /** Returns an opaque identifier of stderr stream for the process. */
  public static native long getStderr(long process);

  /**
   * Returns an estimate of the number of bytes available to read on the stream. Unlike {@link
   * InputStream#available()}, this returns 0 on closed or broken streams.
   */
  public static native int streamBytesAvailable(long stream);

  /**
   * Reads data from the stream into the given array. {@code stream} should come from {@link
   * #getStdout(long)} or {@link #getStderr(long)}.
   *
   * <p>Blocks until either some data was read or the process is terminated.
   *
   * @return the number of bytes read, 0 on EOF, or -1 if there was an error.
   */
  public static native int readStream(long stream, byte[] bytes, int offset, int length);

  /**
   * Waits until the given process terminates. If timeout is non-negative, it indicates the number
   * of milliseconds before the call times out.
   *
   * <p>Return values:
   * <li>0: Process finished
   * <li>1: Timeout
   * <li>2: Something went wrong
   */
  public static native int waitFor(long process, long timeout);

  /**
   * Returns the exit code of the process. Throws {@code IllegalStateException} if something goes
   * wrong.
   */
  public static native int getExitCode(long process);

  /** Returns the process ID of the given process or -1 if there was an error. */
  public static native int getProcessPid(long process);

  /** Terminates the given process. Returns true if the termination was successful. */
  public static native boolean terminate(long process);

  /**
   * Releases the native data structures associated with the process.
   *
   * <p>Calling any other method on the same process after this call will result in the JVM crashing
   * or worse.
   */
  public static native void deleteProcess(long process);

  /**
   * Closes the stream
   *
   * @param stream should come from {@link #getStdout(long)} or {@link #getStderr(long)}.
   */
  public static native void closeStream(long stream);

  /**
   * Returns a string representation of the last error caused by any call on the given process or
   * the empty string if the last operation was successful.
   *
   * <p>Does <b>NOT</b> terminate the process if it is still running.
   *
   * <p>After this call returns, subsequent calls will return the empty string if there was no
   * failed operation in between.
   */
  public static native String processGetLastError(long process);

  public static native String streamGetLastError(long process);

  /** Returns the PID of the current process. */
  public static native int getpid();
}
