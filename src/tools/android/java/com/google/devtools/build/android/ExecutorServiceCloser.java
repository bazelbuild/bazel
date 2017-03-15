// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import com.google.common.util.concurrent.ListeningExecutorService;
import java.io.Closeable;
import java.io.IOException;
import java.util.List;

/** Shutdowns and verifies that no tasks are running in the executor service. */
final class ExecutorServiceCloser implements Closeable {
  private final ListeningExecutorService executorService;

  private ExecutorServiceCloser(ListeningExecutorService executorService) {
    this.executorService = executorService;
  }

  @Override
  public void close() throws IOException {
    List<Runnable> unfinishedTasks = executorService.shutdownNow();
    if (!unfinishedTasks.isEmpty()) {
      throw new IOException("Shutting down the executor with unfinished tasks:" + unfinishedTasks);
    }
  }

  public static Closeable createWith(ListeningExecutorService executorService) {
    return new ExecutorServiceCloser(executorService);
  }
}
