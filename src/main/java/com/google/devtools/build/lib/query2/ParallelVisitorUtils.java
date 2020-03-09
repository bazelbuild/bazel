// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2;

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.concurrent.BlockingStack;
import com.google.devtools.build.lib.concurrent.ParallelVisitor;
import com.google.devtools.build.lib.concurrent.ParallelVisitor.Factory;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/** Utilities for {@link ParallelVisitor} with QueryException/Target type parameters. */
public final class ParallelVisitorUtils {

  /** All visitors share a single global fixed thread pool. */
  static final ExecutorService FIXED_THREAD_POOL_EXECUTOR =
      new ThreadPoolExecutor(
          /*corePoolSize=*/ Math.max(1, SkyQueryEnvironment.DEFAULT_THREAD_COUNT),
          /*maximumPoolSize=*/ Math.max(1, SkyQueryEnvironment.DEFAULT_THREAD_COUNT),
          /*keepAliveTime=*/ 1,
          /*units=*/ TimeUnit.SECONDS,
          /*workQueue=*/ new BlockingStack<>(),
          new ThreadFactoryBuilder().setNameFormat("parallel-visitor %d").build());

  /**
   * Returns a {@link Callback} which kicks off a parallel visitation when {@link Callback#process}
   * is invoked.
   */
  public static <OutputResultT extends Target, CallbackT extends Callback<OutputResultT>>
      Callback<OutputResultT> createParallelVisitorCallback(
          Factory<SkyKey, ?, ?, OutputResultT, QueryException, CallbackT> visitorFactory) {
    return new ParallelTargetVisitorCallback<>(visitorFactory);
  }

  /** Factory for creating ParallelVisitors used during Query execution. */
  public interface QueryVisitorFactory<VisitKeyT, OutputKeyT, OutputResultT>
      extends Factory<
          SkyKey, VisitKeyT, OutputKeyT, OutputResultT, QueryException, Callback<OutputResultT>> {}

  /**
   * A {@link Callback} whose {@link Callback#process} method kicks off a visitation via a fresh
   * {@link ParallelVisitor} instance.
   */
  public static class ParallelTargetVisitorCallback<
          OutputResultT extends Target, CallbackT extends Callback<OutputResultT>>
      implements Callback<OutputResultT> {
    private final ParallelVisitor.Factory<SkyKey, ?, ?, OutputResultT, QueryException, CallbackT>
        visitorFactory;

    public ParallelTargetVisitorCallback(
        ParallelVisitor.Factory<SkyKey, ?, ?, OutputResultT, QueryException, CallbackT>
            visitorFactory) {
      this.visitorFactory = visitorFactory;
    }

    @Override
    public void process(Iterable<OutputResultT> partialResult)
        throws QueryException, InterruptedException {
      ParallelVisitor<SkyKey, ?, ?, OutputResultT, QueryException, CallbackT> visitor =
          visitorFactory.create();
      // TODO(b/131109214): It's not ideal to have an operation like this in #process that blocks on
      // another, potentially expensive computation. Refactor to something like "processAsync".
      visitor.visitAndWaitForCompletion(SkyQueryEnvironment.makeLabelsStrict(partialResult));
    }
  }

  /** A ParallelVisitor suitable for use during query execution. */
  public abstract static class ParallelQueryVisitor<VisitKeyT, OutputKeyT, OutputResultT>
      extends ParallelVisitor<
          SkyKey, VisitKeyT, OutputKeyT, OutputResultT, QueryException, Callback<OutputResultT>> {
    public ParallelQueryVisitor(
        Callback<OutputResultT> callback,
        int visitBatchSize,
        int processResultsBatchSize,
        VisitTaskStatusCallback visitTaskStatusCallback) {
      super(
          callback,
          QueryException.class,
          visitBatchSize,
          processResultsBatchSize,
          3L * SkyQueryEnvironment.DEFAULT_THREAD_COUNT,
          SkyQueryEnvironment.BATCH_CALLBACK_SIZE,
          FIXED_THREAD_POOL_EXECUTOR,
          visitTaskStatusCallback);
    }
  }
}
