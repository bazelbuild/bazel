// Copyright 2018 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.Uniquifier;
import com.google.devtools.build.skyframe.SkyKey;

/**
 * Helper class for visiting the TTV-only DTC of some given TTV keys, via BFS following all
 * TTV->TTV dep edges. Disallowed edge filtering is *not* performed.
 */
public abstract class AbstractUnfilteredTTVDTCVisitor<T> extends AbstractSkyKeyParallelVisitor<T> {
  protected final SkyQueryEnvironment env;

  protected AbstractUnfilteredTTVDTCVisitor(
      SkyQueryEnvironment env,
      Uniquifier<SkyKey> uniquifier,
      int processResultsBatchSize,
      Callback<T> callback) {
    super(
        uniquifier,
        callback,
        env.getVisitBatchSizeForParallelVisitation(),
        processResultsBatchSize);
    this.env = env;
  }

  @Override
  protected Visit getVisitResult(Iterable<SkyKey> ttvKeys) throws InterruptedException {
    Multimap<SkyKey, SkyKey> deps = env.getUnfilteredDirectDepsOfSkyKeys(ttvKeys);
    return new Visit(
        /*keysToUseForResult=*/ deps.keySet(),
        /*keysToVisit=*/ deps.values()
        .stream()
        .filter(SkyQueryEnvironment.IS_TTV)
        .collect(ImmutableList.toImmutableList()));
  }

  @Override
  protected Iterable<SkyKey> preprocessInitialVisit(Iterable<SkyKey> visitationKeys) {
    // ParallelVisitorCallback passes in TTV keys.
    Preconditions.checkState(
        Iterables.all(visitationKeys, SkyQueryEnvironment.IS_TTV), visitationKeys);
    return visitationKeys;
  }
}
