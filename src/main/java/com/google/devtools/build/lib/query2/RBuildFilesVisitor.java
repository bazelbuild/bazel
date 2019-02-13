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

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpressionContext;
import com.google.devtools.build.lib.query2.engine.Uniquifier;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Collection;
import java.util.Set;
import java.util.function.Function;

/** A helper class that computes 'rbuildfiles(<blah>)' via BFS. */
public class RBuildFilesVisitor extends AbstractSkyKeyParallelVisitor<Target> {

  // Each target in the full output of 'rbuildfiles' corresponds to BUILD file InputFile of a
  // unique package. So the processResultsBatchSize we choose to pass to the ParallelVisitor ctor
  // influences how many packages each leaf task doing processPartialResults will have to
  // deal with at once. A value of 100 was chosen experimentally.
  private static final int PROCESS_RESULTS_BATCH_SIZE = 100;
  private static final SkyKey EXTERNAL_PACKAGE_KEY =
      PackageValue.key(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER);
  private final SkyQueryEnvironment env;
  private final QueryExpressionContext<Target> context;
  private final Function<SkyKey, Boolean> rdepFilter;

  public RBuildFilesVisitor(
      SkyQueryEnvironment env,
      Uniquifier<SkyKey> uniquifier,
      QueryExpressionContext<Target> context,
      Callback<Target> callback,
      Function<SkyKey, Boolean> rdepFilter) {
    super(uniquifier, callback, ParallelSkyQueryUtils.VISIT_BATCH_SIZE, PROCESS_RESULTS_BATCH_SIZE);
    this.env = env;
    this.context = context;
    this.rdepFilter = rdepFilter;
  }

  @Override
  protected Visit getVisitResult(Iterable<SkyKey> values) throws InterruptedException {
    Collection<Iterable<SkyKey>> reverseDeps = env.graph.getReverseDeps(values).values();
    Set<SkyKey> keysToUseForResult = CompactHashSet.create();
    Set<SkyKey> keysToVisitNext = CompactHashSet.create();
    for (SkyKey rdep : Iterables.concat(reverseDeps)) {
      if (rdep.functionName().equals(SkyFunctions.PACKAGE)) {
        keysToUseForResult.add(rdep);
        // Every package has a dep on the external package, so we need to include those edges too.
        if (rdep.equals(EXTERNAL_PACKAGE_KEY)) {
          keysToVisitNext.add(rdep);
        }
      } else if (rdepFilter.apply(rdep)) {
        keysToVisitNext.add(rdep);
      }
    }
    return new Visit(keysToUseForResult, keysToVisitNext);
  }

  @Override
  protected void processPartialResults(
      Iterable<SkyKey> keysToUseForResult, Callback<Target> callback)
      throws QueryException, InterruptedException {
    env.getBuildFileTargetsForPackageKeysAndProcessViaCallback(
        keysToUseForResult, context, callback);
  }

  @Override
  protected Iterable<SkyKey> preprocessInitialVisit(Iterable<SkyKey> keys) {
    return keys;
  }
}
