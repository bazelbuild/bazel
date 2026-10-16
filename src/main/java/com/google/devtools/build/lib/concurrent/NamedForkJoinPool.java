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
package com.google.devtools.build.lib.concurrent;

import java.lang.Thread.UncaughtExceptionHandler;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinWorkerThread;
import java.util.concurrent.atomic.AtomicLong;
import javax.annotation.Nullable;

/** A {@link ForkJoinPool} with support for thread naming. */
public class NamedForkJoinPool extends ForkJoinPool {

  /**
   * Creates a new NamedForkJoinPool.
   *
   * @param name A string identifying the pool. This string must not contain any formatting
   *     parameters.
   * @param numThreads The maximum number of threads to create, see {@link ForkJoinPool}.
   */
  public static NamedForkJoinPool newNamedPool(String name, int numThreads) {
    return newNamedPool(name, numThreads, null);
  }

  /**
   * Creates a new NamedForkJoinPool with an explicit uncaught exception handler.
   *
   * @param name A string identifying the pool. This string must not contain any formatting
   *     parameters.
   * @param numThreads The maximum number of threads to create, see {@link ForkJoinPool}.
   * @param uncaughtExceptionHandler The handler for uncaught exceptions in worker threads.
   */
  public static NamedForkJoinPool newNamedPool(
      String name, int numThreads, @Nullable UncaughtExceptionHandler uncaughtExceptionHandler) {
    return new NamedForkJoinPool(name, numThreads, uncaughtExceptionHandler);
  }

  private NamedForkJoinPool(
      String name, int poolSize, @Nullable UncaughtExceptionHandler uncaughtExceptionHandler) {
    super(
        poolSize,
        new NamedForkJoinWorkerThreadFactory(name + "-%s"),
        uncaughtExceptionHandler,
        /* asyncMode= */ false);
  }

  /** A {@link ForkJoinWorkerThread} named on construction. */
  private static class NamedForkJoinWorkerThread extends ForkJoinWorkerThread {

    public NamedForkJoinWorkerThread(ForkJoinPool forkJoinPool, String name) {
      super(forkJoinPool);
      this.setName(name);
    }
  }

  /**
   * A factory for {@link NamedForkJoinWorkerThread}s that names those threads using a
   * client-provided name format that consumes a thread index.
   */
  private static class NamedForkJoinWorkerThreadFactory implements ForkJoinWorkerThreadFactory {

    private final String nameFormat;
    private final AtomicLong nextUnusedThreadIndex = new AtomicLong(0L);

    NamedForkJoinWorkerThreadFactory(String nameFormat) {
      this.nameFormat = nameFormat;
    }

    @Override
    public ForkJoinWorkerThread newThread(ForkJoinPool forkJoinPool) {
      return new NamedForkJoinWorkerThread(
          forkJoinPool, String.format(nameFormat, nextUnusedThreadIndex.getAndIncrement()));
    }
  }
}
