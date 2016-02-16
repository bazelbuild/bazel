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

package com.google.devtools.build.singlejar;

import com.google.devtools.build.singlejar.ZipEntryFilter.CustomMergeStrategy;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import javax.annotation.concurrent.NotThreadSafe;

/**
 * A strategy that merges a set of files by concatenating them. This is used
 * for services files. By default, this class automatically adds a newline
 * character {@code '\n'} between files if the previous file did not end with one.
 *
 * <p>Note: automatically inserting newline characters differs from the
 * original behavior. Use {@link #ConcatenateStrategy(boolean)} to turn this
 * behavior off.
 */
@NotThreadSafe
public final class ConcatenateStrategy implements CustomMergeStrategy {

  // The strategy assumes that files are generally small. This is a first guess
  // about the size of the files.
  private static final int BUFFER_SIZE = 4096;

  private final byte[] buffer = new byte[BUFFER_SIZE];
  private byte lastByteCopied = '\n';
  private final boolean appendNewLine;

  public ConcatenateStrategy() {
    this(true);
  }

  /**
   * @param appendNewLine Whether to add a newline character between files if
   *                      the previous file did not end with one.
   */
  public ConcatenateStrategy(boolean appendNewLine) {
    this.appendNewLine = appendNewLine;
  }

  @Override
  public void merge(InputStream in, OutputStream out) throws IOException {
    if (appendNewLine && lastByteCopied != '\n') {
      out.write('\n');
      lastByteCopied = '\n';
    }
    int bytesRead;
    while ((bytesRead = in.read(buffer)) != -1) {
      out.write(buffer, 0, bytesRead);
      lastByteCopied = buffer[bytesRead - 1];
    }
  }

  @Override
  public void finish(OutputStream out) {
    // No need to do anything. All the data was already written.
  }
}
