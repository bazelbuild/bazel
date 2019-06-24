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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.query2.engine.QueryUtil.AggregateAllCallback;
import com.google.devtools.build.lib.query2.engine.Uniquifier;
import com.google.devtools.build.skyframe.SkyKey;

/**
 * Helper class for visiting the TTV-only DTC of some given TTV keys, and feeding those TTVs to a
 * callback.
 */
class UnfilteredSkyKeyTTVDTCVisitor extends AbstractUnfilteredTTVDTCVisitor<SkyKey> {
  private UnfilteredSkyKeyTTVDTCVisitor(
      SkyQueryEnvironment env,
      Uniquifier<SkyKey> uniquifier,
      int processResultsBatchSize,
      AggregateAllCallback<SkyKey, ImmutableSet<SkyKey>> aggregateAllCallback) {
    super(
        env,
        uniquifier,
        processResultsBatchSize,
        aggregateAllCallback);
  }

  @Override
  protected Iterable<SkyKey> outputKeysToOutputValues(Iterable<SkyKey> targetKeys) {
    return targetKeys;
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
    public ParallelVisitor<SkyKey, SkyKey, SkyKey> create() {
      return new UnfilteredSkyKeyTTVDTCVisitor(
          env, uniquifier, processResultsBatchSize, aggregateAllCallback);
    }
  }
}
