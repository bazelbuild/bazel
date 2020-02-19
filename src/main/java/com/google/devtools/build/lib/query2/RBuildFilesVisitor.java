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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.ParallelVisitorUtils.ParallelQueryVisitor;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpressionContext;
import com.google.devtools.build.lib.query2.engine.Uniquifier;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Collection;
import java.util.Set;

/** A helper class that computes 'rbuildfiles(<blah>)' via BFS. */
public class RBuildFilesVisitor extends ParallelQueryVisitor<SkyKey, PackageIdentifier, Target> {

  // Each target in the full output of 'rbuildfiles' corresponds to BUILD file InputFile of a
  // unique package. So the processResultsBatchSize we choose to pass to the ParallelVisitor ctor
  // influences how many packages each leaf task doing processPartialResults will have to
  // deal with at once. A value of 100 was chosen experimentally.
  private static final int PROCESS_RESULTS_BATCH_SIZE = 100;

  // We don't expect to find any additional BUILD files so we skip visitation of the following
  // nodes.
  private static final ImmutableSet<SkyFunctionName> NODES_TO_PRUNE_TRAVERSAL =
      ImmutableSet.of(
          Label.TRANSITIVE_TRAVERSAL,
          SkyFunctions.COLLECT_TARGETS_IN_PACKAGE,
          SkyFunctions.COLLECT_TEST_SUITES_IN_PACKAGE,
          SkyFunctions.PREPARE_DEPS_OF_TARGETS_UNDER_DIRECTORY,
          SkyFunctions.PREPARE_TEST_SUITES_UNDER_DIRECTORY,
          SkyFunctions.PACKAGE_ERROR_MESSAGE,
          SkyFunctions.PREPARE_DEPS_OF_PATTERN,
          SkyFunctions.PREPARE_DEPS_OF_PATTERNS);

  private static final SkyKey EXTERNAL_PACKAGE_KEY =
      PackageValue.key(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER);
  private final SkyQueryEnvironment env;
  private final QueryExpressionContext<Target> context;
  private final Uniquifier<SkyKey> visitUniquifier;
  protected final Uniquifier<SkyKey> resultUniquifier;

  public RBuildFilesVisitor(
      SkyQueryEnvironment env,
      Uniquifier<SkyKey> visitUniquifier,
      Uniquifier<SkyKey> resultUniquifier,
      QueryExpressionContext<Target> context,
      Callback<Target> callback) {
    super(
        callback,
        env.getVisitBatchSizeForParallelVisitation(),
        PROCESS_RESULTS_BATCH_SIZE,
        env.getVisitTaskStatusCallback());
    this.env = env;
    this.visitUniquifier = visitUniquifier;
    this.resultUniquifier = resultUniquifier;
    this.context = context;
  }

  @Override
  protected Visit getVisitResult(Iterable<SkyKey> values)
      throws QueryException, InterruptedException {
    Collection<Iterable<SkyKey>> reverseDeps = env.graph.getReverseDeps(values).values();
    Set<PackageIdentifier> keysToUseForResult = CompactHashSet.create();
    Set<SkyKey> keysToVisitNext = CompactHashSet.create();
    for (SkyKey rdep : Iterables.concat(reverseDeps)) {
      if (rdep.functionName().equals(SkyFunctions.PACKAGE)) {
        if (resultUniquifier.unique(rdep)) {
          keysToUseForResult.add((PackageIdentifier) rdep.argument());
        }
        // PackageValue(//p) has a transitive dep on the PackageValue(//external), so we need to
        // make sure these dep paths are traversed. These dep paths go through the singleton
        // WorkspaceNameValue(), and that node has a direct dep on PackageValue(//external), so it
        // suffices to ensure we visit PackageValue(//external).
        if (rdep.equals(EXTERNAL_PACKAGE_KEY)) {
          keysToVisitNext.add(rdep);
        }
      } else if (!NODES_TO_PRUNE_TRAVERSAL.contains(rdep.functionName())) {
        processNonPackageRdepAndDetermineVisitations(rdep, keysToVisitNext, keysToUseForResult);
      }
    }
    return new Visit(keysToUseForResult, keysToVisitNext);
  }

  @Override
  protected Iterable<SkyKey> preprocessInitialVisit(Iterable<SkyKey> visitationKeys) {
    return visitationKeys;
  }

  protected void processNonPackageRdepAndDetermineVisitations(
      SkyKey rdep, Set<SkyKey> keysToVisitNext, Set<PackageIdentifier> keysToUseForResult)
      throws QueryException {
    // Packages may depend on the existence of subpackages, but these edges aren't
    // relevant to rbuildfiles. They may also depend on files transitively through
    // globs, but these cannot be included in load statements and so we don't traverse
    // through these either.
    if (!rdep.functionName().equals(SkyFunctions.PACKAGE_LOOKUP)
        && !rdep.functionName().equals(SkyFunctions.GLOB)) {
      keysToVisitNext.add(rdep);
    }
  }

  @Override
  protected Iterable<Target> outputKeysToOutputValues(Iterable<PackageIdentifier> targetKeys)
      throws QueryException, InterruptedException {
    return env.getBuildFileTargetsForPackageKeys(ImmutableSet.copyOf(targetKeys), context);
  }

  @Override
  protected Iterable<SkyKey> noteAndReturnUniqueVisitationKeys(
      Iterable<SkyKey> prospectiveVisitationKeys) throws QueryException {
    return visitUniquifier.unique(prospectiveVisitationKeys);
  }
}
