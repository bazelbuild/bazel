// Copyright 2026 The Bazel Authors. All rights reserved.
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
package net.starlark.java.eval;

import java.io.FileDescriptor;

/** Provides access to the native CPU profiler implementation. */
@SuppressWarnings("GoodTime")
public interface CpuProfilerNativeSupport {

  /**
   * Returns whether CPU profiling is supported on this platform.
   *
   * <p>Other methods may not be called if this returns false.
   */
  boolean supported();

  /**
   * Returns the read end of a pipe from which profiling events may be read. Each event is an
   * operating system thread ID as returned by {@code getThreadId()}, encoded as a big-endian 32-bit
   * integer, indicating that a time quantum has elapsed for the corresponding thread.
   *
   * <p>Must not be called more than once.
   */
  FileDescriptor createPipe();

  /**
   * Starts the operating system's interval timer.
   *
   * <p>The period must be a positive number of microseconds.
   *
   * <p>Returns false if a SIGPROF signal handler is already set, including by an earlier call to
   * {@link #startTimer} that hasn't been followed by {@link #stopTimer}.
   */
  boolean startTimer(long periodMicros);

  /** Stops the operating system's interval timer. */
  void stopTimer();

  /** Returns the operating system's identifier for the calling thread. */
  int getThreadId();
}
