// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.EventHandler;

import java.io.PrintStream;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

import javax.annotation.Nullable;

/**
 * A poorly-performing but functioning versioned evaluator. This evaluator stores all past
 * evaluation from previous versions.
 *
 * <p>Current limitations are plentiful: There is no sharing of state across versions. There is no
 * GC policy. Node injection doesn't work.
 */
public final class TrivialVersionedEvaluator implements MemoizingEvaluator {

  private final ImmutableMap<? extends SkyFunctionName, ? extends SkyFunction> skyFunctions;
  @Nullable private final EvaluationProgressReceiver progressReceiver;

  private final LoadingCache<Version, InMemoryGraph> versionedGraphs;
  private final EmittedEventState emittedEventState;
  private final AtomicBoolean evaluating = new AtomicBoolean(false);

  public TrivialVersionedEvaluator(
      Map<? extends SkyFunctionName, ? extends SkyFunction> skyFunctions,
      @Nullable EvaluationProgressReceiver invalidationReceiver,
      EmittedEventState emittedEventState, boolean keepEdges) {
    this.skyFunctions = ImmutableMap.copyOf(skyFunctions);
    this.progressReceiver = invalidationReceiver;
    this.versionedGraphs = CacheBuilder.newBuilder().build(
        new CacheLoader<Version, InMemoryGraph>() {
          @Override
          public InMemoryGraph load(Version version) {
            return new InMemoryGraph(/*keepEdges=*/true);
          }
        });
    this.emittedEventState = emittedEventState;
  }

  @Override
  public <T extends SkyValue> EvaluationResult<T> evaluate(Iterable<SkyKey> roots, Version version,
          boolean keepGoing, int numThreads, EventHandler eventHandler)
      throws InterruptedException {
    // TODO(bazel-team): Do not assume Integer versioning. Replace or otherwise deal with
    // existing ParallelEvaluator requirement of passing in an integer version number.
    // [skyframe-versioning]
    IntVersion intVersion = (IntVersion) version;

    // TODO(bazel-team): Allow for greater concurrency. First, allow for multiple versions to be
    // evaluated in parallel. Next, allow for intra-version parallelism. [skyframe-versioning]
    setAndCheckEvaluateState(true, roots);
    try {
      // TODO(bazel-team): Deal with Skyframe injection. [skyframe-versioning]
      return new ParallelEvaluator(versionedGraphs.getUnchecked(version), intVersion.getVal(),
          skyFunctions, eventHandler, emittedEventState, keepGoing, numThreads,
          progressReceiver).eval(roots);
    } finally {
      setAndCheckEvaluateState(false, roots);
    }
  }

  private void setAndCheckEvaluateState(boolean newValue, Object requestInfo) {
    Preconditions.checkState(evaluating.getAndSet(newValue) != newValue,
        "Re-entrant evaluation for request: %s", requestInfo);
  }

  @Override
  public void delete(final Predicate<SkyKey> deletePredicate) {
    // TODO(bazel-team): Deletion is not going to work in versioned/shared evaluation.
    // [skyframe-versioning]
  }

  @Override
  public void deleteDirty(long versionAgeLimit) {
    // Do nothing.
  }

  @Override
  public Map<SkyKey, SkyValue> getValues() {
    // TODO(bazel-team): Avoid full graph iteration in versioned Skyframe. [skyframe-versioning]
    return ImmutableMap.of();
  }

  @Override
  public Map<SkyKey, SkyValue> getDoneValues() {
    // TODO(bazel-team): Avoid full graph iteration in versioned Skyframe. [skyframe-versioning]
    throw unsupportedOnTrivialVersioning();
  }

  @Override
  @Nullable public SkyValue getExistingValueForTesting(SkyKey key) {
    // TODO(bazel-team): Support testing lookups. [skyframe-versioning]
    throw unsupportedOnTrivialVersioning();
  }

  @Override
  @Nullable public ErrorInfo getExistingErrorForTesting(SkyKey key) {
    // TODO(bazel-team): Support testing lookups. [skyframe-versioning]
    throw unsupportedOnTrivialVersioning();
  }

  @Override
  public void dump(PrintStream out) {
    // TODO(bazel-team): Support dumping some information about versioned Skyframe.
    // [skyframe-versioning]
    unsupportedOnTrivialVersioning();
  }

  private static RuntimeException unsupportedOnTrivialVersioning() {
    throw new UnsupportedOperationException("Unsupported in current trivial versioning");
  }
}
