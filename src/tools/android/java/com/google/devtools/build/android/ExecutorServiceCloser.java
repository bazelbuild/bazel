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
import com.google.common.util.concurrent.MoreExecutors;
import java.io.Closeable;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.Executors;

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
      throw new IOException(
          "Shutting down the executor with unfinished tasks:" + unfinishedTasks.size());
    }
  }

  public static Closeable createWith(ListeningExecutorService executorService) {
    return new ExecutorServiceCloser(executorService);
  }

  /**
   * Creates a {@link ListeningExecutorService} with a sane sized thread pool based on our current
   * metrics.
   */
  public static ListeningExecutorService createDefaultService() {
    // The reported availableProcessors may be higher than the actual resources
    // (on a shared system). On the other hand, a lot of the work is I/O, so it's not completely
    // CPU bound. As a compromise, divide by 2 the reported availableProcessors.
    int numThreads = Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
    final ListeningExecutorService executorService =
        MoreExecutors.listeningDecorator(Executors.newFixedThreadPool(numThreads));
    return executorService;
  }
}
