// Copyright 2020 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.exec;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.io.ByteStreams;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.util.io.FileWatcher;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.Path;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.TimeUnit;

/** Implements the --test_output=streamed option. */
public class StreamedTestOutput implements Closeable {
  private static final int JOIN_ON_INTERRUPT_GRACE_PERIOD_SECONDS = 30;

  private final TestLogHelper.FilterTestHeaderOutputStream headerFilter;
  private final FileWatcher watcher;
  private final Path testLogPath;
  private final OutErr outErr;

  public StreamedTestOutput(OutErr outErr, Path testLogPath) throws IOException {
    this.testLogPath = testLogPath;
    this.outErr = outErr;
    this.headerFilter = TestLogHelper.getHeaderFilteringOutputStream(outErr.getOutputStream());
    this.watcher = new FileWatcher(testLogPath, OutErr.create(headerFilter, headerFilter), false);
    watcher.start();
  }

  @Override
  public void close() throws IOException {
    watcher.stopPumping();
    try {
      // The watcher thread might leak if the following call is interrupted.
      // This is a relatively minor issue since the worst it could do is
      // write one additional line from the test.log to the console later on
      // in the build.
      watcher.join();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      watcher.interrupt();
      Uninterruptibles.joinUninterruptibly(
          watcher, JOIN_ON_INTERRUPT_GRACE_PERIOD_SECONDS, TimeUnit.SECONDS);
      Preconditions.checkState(
          !watcher.isAlive(),
          "Watcher thread failed to exit for %s seconds after interrupt",
          JOIN_ON_INTERRUPT_GRACE_PERIOD_SECONDS);
    }

    // It's unclear if writing this after interrupt is desirable, but it's been this way forever.
    if (!headerFilter.foundHeader()) {
      try (InputStream input = testLogPath.getInputStream()) {
        ByteStreams.copy(input, outErr.getOutputStream());
      }
    }
  }

  @VisibleForTesting
  FileWatcher getFileWatcher() {
    return watcher;
  }
}
