// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.shell;

/**
 * Supplier of the command result which additionally allows to check if the command already
 * terminated. Implementations of this interface may not be thread-safe.
 */
public interface FutureCommandResult {
  /**
   * Returns the result of command execution. If the process is not finished yet (as reported by
   * {@link #isDone()}, the call will block until that process terminates.
   *
   * @return non-null result of command execution
   * @throws AbnormalTerminationException if command execution failed
   */
  CommandResult get() throws AbnormalTerminationException;

  /**
   * Aborts the subprocess if it is still running. Note that it does not immediately terminate the
   * process, so {@link #isDone} may still return true if called immediately afterwards.
   */
  void cancel();

  /**
   * Returns true if the process terminated, the command result is available and the call to
   * {@link #get()} will not block.
   *
   * @return true if the process terminated
   */
  boolean isDone();
}
