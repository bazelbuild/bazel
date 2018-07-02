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
package com.google.devtools.build.lib.query2;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryUtil.AggregateAllCallback;
import com.google.devtools.build.lib.query2.engine.Uniquifier;
import com.google.devtools.build.skyframe.SkyKey;

/**
 * Helper class that computes the TTV-only DTC of some given TTV keys, via BFS following all
 * TTV->TTV dep edges.
 */
class TransitiveTraversalValueDTCVisitor extends ParallelVisitor<SkyKey, SkyKey> {
  private final SkyQueryEnvironment env;
  private final Uniquifier<SkyKey> uniquifier;

  private TransitiveTraversalValueDTCVisitor(
      SkyQueryEnvironment env,
      Uniquifier<SkyKey> uniquifier,
      int processResultsBatchSize,
      AggregateAllCallback<SkyKey, ImmutableSet<SkyKey>> aggregateAllCallback) {
    super(aggregateAllCallback, ParallelSkyQueryUtils.VISIT_BATCH_SIZE, processResultsBatchSize);
    this.env = env;
    this.uniquifier = uniquifier;
  }

  static class Factory implements ParallelVisitor.Factory {
    private final SkyQueryEnvironment env;
    private final Uniquifier<SkyKey> uniquifier;
    private final AggregateAllCallback<SkyKey, ImmutableSet<SkyKey>> aggregateAllCallback;
    private final int processResultsBatchSize;

    Factory(
        SkyQueryEnvironment env,
        Uniquifier<SkyKey> uniquifier,
        int processResultsBatchSize,
        AggregateAllCallback<SkyKey, ImmutableSet<SkyKey>> aggregateAllCallback) {
      this.env = env;
      this.uniquifier = uniquifier;
      this.processResultsBatchSize = processResultsBatchSize;
      this.aggregateAllCallback = aggregateAllCallback;
    }

    @Override
    public ParallelVisitor<SkyKey, SkyKey> create() {
      return new TransitiveTraversalValueDTCVisitor(
          env, uniquifier, processResultsBatchSize, aggregateAllCallback);
    }
  }

  @Override
  protected void processPartialResults(
      Iterable<SkyKey> keysToUseForResult, Callback<SkyKey> callback)
      throws QueryException, InterruptedException {
    callback.process(keysToUseForResult);
  }

  @Override
  protected Visit getVisitResult(Iterable<SkyKey> ttvKeys) throws InterruptedException {
    Multimap<SkyKey, SkyKey> deps = env.getDirectDepsOfSkyKeys(ttvKeys);
    return new Visit(
        /*keysToUseForResult=*/ deps.keySet(),
        /*keysToVisit=*/ deps.values()
            .stream()
            .filter(SkyQueryEnvironment.IS_TTV)
            .collect(ImmutableList.toImmutableList()));
  }

  @Override
  protected Iterable<SkyKey> preprocessInitialVisit(Iterable<SkyKey> keys) {
    // ParallelVisitorCallback passes in TTV keys.
    Preconditions.checkState(Iterables.all(keys, SkyQueryEnvironment.IS_TTV), keys);
    return keys;
  }

  @Override
  protected ImmutableList<SkyKey> getUniqueValues(Iterable<SkyKey> values) throws QueryException {
    return uniquifier.unique(values);
  }
}
