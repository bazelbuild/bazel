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

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.MultisetSemaphore;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.ParallelVisitorUtils.ParallelQueryVisitor;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Collection;
import java.util.Map;
import java.util.Set;

/**
 * Helper class to traverse a visitation graph where the outputs are {@link Target}s and there is a
 * simple mapping between visitation keys and output keys.
 */
public abstract class AbstractTargetOuputtingVisitor<VisitKeyT>
    extends ParallelQueryVisitor<VisitKeyT, SkyKey, Target> {
  private static final int PROCESS_RESULTS_BATCH_SIZE = SkyQueryEnvironment.BATCH_CALLBACK_SIZE;

  protected final SkyQueryEnvironment env;

  protected AbstractTargetOuputtingVisitor(SkyQueryEnvironment env, Callback<Target> callback) {
    super(
        callback,
        env.getVisitBatchSizeForParallelVisitation(),
        PROCESS_RESULTS_BATCH_SIZE,
        env.getVisitTaskStatusCallback());
    this.env = env;
  }

  @Override
  protected Iterable<Target> outputKeysToOutputValues(Iterable<SkyKey> targetKeys)
      throws InterruptedException, QueryException {
    Map<Label, Target> targets =
        env.getTargets(Iterables.transform(targetKeys, SkyQueryEnvironment.SKYKEY_TO_LABEL));

    handleMissingTargets(targets, ImmutableSet.copyOf(targetKeys));
    return targets.values();
  }

  void handleMissingTargets(Map<? extends SkyKey, Target> keysWithTargets, Set<SkyKey> targetKeys)
      throws InterruptedException, QueryException {
    // Do nothing by default, as an optimization if we don't expect any missing targets.
  }

  @Override
  protected Iterable<Task<QueryException>> getVisitTasks(Collection<VisitKeyT> pendingVisits)
      throws InterruptedException, QueryException {
    // Group pending visitation by the package of the new node, since we'll be targetfying the
    // node during the visitation.
    ListMultimap<PackageIdentifier, VisitKeyT> visitsByPackage = ArrayListMultimap.create();
    for (VisitKeyT visitationKey : pendingVisits) {
      // Overrides of visitationKeyToOutputKey are non-blocking.
      SkyKey skyKey = visitationKeyToOutputKey(visitationKey);
      if (skyKey != null) {
        Label label = SkyQueryEnvironment.SKYKEY_TO_LABEL.apply(skyKey);
        visitsByPackage.put(label.getPackageIdentifier(), visitationKey);
      }
    }

    ImmutableList.Builder<Task<QueryException>> builder = ImmutableList.builder();

    // A couple notes here:
    // (i)  ArrayListMultimap#values returns the values grouped by key, which is exactly what we
    //      want.
    // (ii) ArrayListMultimap#values returns a Collection view, so we make a copy to avoid
    //      accidentally retaining the entire ArrayListMultimap object.
    for (Iterable<VisitKeyT> visitBatch :
        Iterables.partition(
            ImmutableList.copyOf(visitsByPackage.values()),
            ParallelSkyQueryUtils.VISIT_BATCH_SIZE)) {
      builder.add(new VisitTask(visitBatch, QueryException.class));
    }

    return builder.build();
  }

  MultisetSemaphore<PackageIdentifier> getPackageSemaphore() {
    return env.getPackageMultisetSemaphore();
  }

  protected abstract SkyKey visitationKeyToOutputKey(VisitKeyT visitationKey)
      throws QueryException, InterruptedException;
}
