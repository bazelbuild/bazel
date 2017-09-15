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

package com.google.devtools.build.lib.shell;

import java.io.Closeable;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * A process started by Bazel.
 */
public interface Subprocess extends Closeable {

  /**
   * Kill the process.
   */
  boolean destroy();

  /**
   * Returns the exit value of the process.
   *
   * <p>Throws {@code IOException} if the process has not terminated yet.
   */
  int exitValue();

  /**
   * Returns the if the process has finished.
   */
  boolean finished();

  /**
   * Returns if the process timed out.
   */
  boolean timedout();

  /**
   * Waits for the process to finish.
   */
  void waitFor() throws InterruptedException;

  /**
   * Returns a stream into which data can be written that the process will get on its stdin.
   */
  OutputStream getOutputStream();

  /**
   * Returns a stream from which the stdout of the process can be read.
   */
  InputStream getInputStream();

  /**
   * Returns a stream from which the stderr of the process can be read.
   */
  InputStream getErrorStream();

  /*
   * Terminates the process as thoroughly as the underlying implementation allows and releases
   * native data structures associated with the process.
   */
  @Override
  void close();
}
