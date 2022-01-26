// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.pkgcache;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.skyframe.TransitiveTargetKey;
import com.google.devtools.build.lib.skyframe.TransitiveTargetValue;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * Preloads transitive packages for query: prepopulates Skyframe with {@link TransitiveTargetValue}
 * objects for the transitive closure of requested targets. To be used when doing a large traversal
 * that benefits from loading parallelism.
 */
public class QueryTransitivePackagePreloader {
  private final Supplier<MemoizingEvaluator> memoizingEvaluatorSupplier;
  private final Supplier<EvaluationContext.Builder> evaluationContextBuilderSupplier;

  public QueryTransitivePackagePreloader(
      Supplier<MemoizingEvaluator> memoizingEvaluatorSupplier,
      Supplier<EvaluationContext.Builder> evaluationContextBuilderSupplier) {
    this.memoizingEvaluatorSupplier = memoizingEvaluatorSupplier;
    this.evaluationContextBuilderSupplier = evaluationContextBuilderSupplier;
  }

  /** Loads the specified {@link TransitiveTargetValue}s. */
  public void preloadTransitiveTargets(
      ExtendedEventHandler eventHandler,
      Iterable<Label> labelsToVisit,
      boolean keepGoing,
      int parallelThreads)
      throws InterruptedException {
    List<SkyKey> valueNames = new ArrayList<>();
    for (Label label : labelsToVisit) {
      valueNames.add(TransitiveTargetKey.of(label));
    }
    EvaluationContext evaluationContext =
        evaluationContextBuilderSupplier
            .get()
            .setKeepGoing(keepGoing)
            .setNumThreads(parallelThreads)
            .setEventHandler(eventHandler)
            .setUseForkJoinPool(true)
            .build();
    memoizingEvaluatorSupplier.get().evaluate(valueNames, evaluationContext);
  }
}
