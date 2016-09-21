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

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinWorkerThread;
import java.util.concurrent.atomic.AtomicLong;

/** A {@link ForkJoinPool} with support for thread naming. */
public class NamedForkJoinPool extends ForkJoinPool {

  protected NamedForkJoinPool(String name, int poolSize) {
    super(
        poolSize,
        new NamedForkJoinWorkerThreadFactory(name + "-%s"),
        /*getUncaughtExceptionHandler=*/ null,
        /*asyncMode=*/ false);
  }

  public static NamedForkJoinPool newNamedPool(String name, int numThreads) {
    return new NamedForkJoinPool(name, numThreads);
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

    public NamedForkJoinWorkerThreadFactory(String nameFormat) {
      // Fail fast if nameFormat is bad.
      String ignored = String.format(nameFormat, 0L);
      this.nameFormat = nameFormat;
    }

    @Override
    public ForkJoinWorkerThread newThread(ForkJoinPool forkJoinPool) {
      return new NamedForkJoinWorkerThread(
          forkJoinPool, String.format(nameFormat, nextUnusedThreadIndex.getAndIncrement()));
    }
  }
}
