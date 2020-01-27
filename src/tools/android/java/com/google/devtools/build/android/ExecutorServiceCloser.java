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

import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import java.io.Closeable;
import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/** Shutdowns and verifies that no tasks are running in the executor service. */
final class ExecutorServiceCloser implements Closeable, ListeningExecutorService {
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

  public static ExecutorServiceCloser createWithFixedPoolOf(int numThreads) {
    return new ExecutorServiceCloser(
        MoreExecutors.listeningDecorator(Executors.newFixedThreadPool(numThreads)));
  }

  public static ExecutorServiceCloser createWith(ListeningExecutorService executorService) {
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

  // Delegate methods below

  @Override
  public <T> ListenableFuture<T> submit(Callable<T> task) {
    return executorService.submit(task);
  }

  @Override
  public ListenableFuture<?> submit(Runnable task) {
    return executorService.submit(task);
  }

  @Override
  public <T> ListenableFuture<T> submit(Runnable task, T result) {
    return executorService.submit(task, result);
  }

  @Override
  public <T> List<Future<T>> invokeAll(Collection<? extends Callable<T>> tasks)
      throws InterruptedException {
    return executorService.invokeAll(tasks);
  }

  @Override
  public <T> List<Future<T>> invokeAll(
      Collection<? extends Callable<T>> tasks, long timeout, TimeUnit unit)
      throws InterruptedException {
    return executorService.invokeAll(tasks, timeout, unit);
  }

  @Override
  public void shutdown() {
    executorService.shutdown();
  }

  @Override
  public List<Runnable> shutdownNow() {
    return executorService.shutdownNow();
  }

  @Override
  public boolean isShutdown() {
    return executorService.isShutdown();
  }

  @Override
  public boolean isTerminated() {
    return executorService.isTerminated();
  }

  @Override
  public boolean awaitTermination(long timeout, TimeUnit unit) throws InterruptedException {
    return executorService.awaitTermination(timeout, unit);
  }

  @Override
  public <T> T invokeAny(Collection<? extends Callable<T>> tasks)
      throws InterruptedException, ExecutionException {
    return executorService.invokeAny(tasks);
  }

  @Override
  public <T> T invokeAny(Collection<? extends Callable<T>> tasks, long timeout, TimeUnit unit)
      throws InterruptedException, ExecutionException, TimeoutException {
    return executorService.invokeAny(tasks, timeout, unit);
  }

  @Override
  public void execute(Runnable command) {
    executorService.execute(command);
  }
}
