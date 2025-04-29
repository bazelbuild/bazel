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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.Uniquifier;
import com.google.devtools.build.skyframe.SkyKey;

/**
 * Helper class for visiting the label-only DTC of some given label keys, via BFS following all
 * target label -> target label dep edges. Disallowed edge filtering is *not* performed.
 */
public abstract class AbstractUnfilteredLabelDTCVisitor<T>
    extends AbstractSkyKeyParallelVisitor<T> {
  protected final SkyQueryEnvironment env;

  private final ImmutableSetMultimap<SkyKey, SkyKey> extraGlobalDeps;

  protected AbstractUnfilteredLabelDTCVisitor(
      SkyQueryEnvironment env,
      Uniquifier<SkyKey> uniquifier,
      int processResultsBatchSize,
      ImmutableSetMultimap<SkyKey, SkyKey> extraGlobalDeps,
      Callback<T> callback) {
    super(
        uniquifier,
        callback,
        env.getVisitBatchSizeForParallelVisitation(),
        processResultsBatchSize,
        env.getVisitTaskStatusCallback());
    this.env = env;
    this.extraGlobalDeps = extraGlobalDeps;
  }

  @Override
  protected Visit getVisitResult(Iterable<SkyKey> labelKeys) throws InterruptedException {
    ImmutableMap<SkyKey, Iterable<SkyKey>> depsMap =
        env.getFwdDepLabels(labelKeys, extraGlobalDeps);
    return new Visit(labelKeys, Iterables.concat(depsMap.values()));
  }

  @Override
  protected Iterable<SkyKey> preprocessInitialVisit(Iterable<SkyKey> visitationKeys) {
    // ParallelTargetVisitorCallback passes in labels.
    Preconditions.checkState(
        Iterables.all(visitationKeys, SkyQueryEnvironment.IS_LABEL), visitationKeys);
    return visitationKeys;
  }
}
