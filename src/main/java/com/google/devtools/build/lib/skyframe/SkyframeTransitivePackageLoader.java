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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.skyframe.BuildDriver;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/** Loads transitive packages for skyframe clients. */
class SkyframeTransitivePackageLoader {

  private final Supplier<BuildDriver> buildDriverSupplier;

  // Needs a Supplier because the SkyframeExecutor creates the BuildDriver on demand.
  public SkyframeTransitivePackageLoader(Supplier<BuildDriver> buildDriverSupplier) {
    this.buildDriverSupplier = buildDriverSupplier;
  }

  /** Loads the specified {@link TransitiveTargetValue}s. */
  EvaluationResult<TransitiveTargetValue> loadTransitiveTargets(
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
        EvaluationContext.newBuilder()
            .setKeepGoing(keepGoing)
            .setNumThreads(parallelThreads)
            .setEventHandler(eventHandler)
            .setUseForkJoinPool(true)
            .build();
    return buildDriverSupplier.get().evaluate(valueNames, evaluationContext);
  }
}
