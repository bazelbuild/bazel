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
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.actions.FileStateValue;
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
import com.google.devtools.build.lib.rules.repository.WorkspaceFileHelper;
import com.google.devtools.build.lib.skyframe.ContainingPackageLookupFunction;
import com.google.devtools.build.lib.skyframe.PackageLookupValue;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.WorkspaceNameValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
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

  /** Initiates the graph visitation algorithm seeded by a set of file paths. */
  public void visitFileIdentifiersAndWaitForCompletion(
      WalkableGraph graph, Iterable<PathFragment> fileKeys)
      throws QueryException, InterruptedException {
    visitAndWaitForCompletion(getSkyKeysForFileFragments(graph, fileKeys));
  }

  /**
   * The passed in {@link PathFragment}s can be associated uniquely to a {@link FileStateValue} with
   * the following logic (the same logic that's in {@link ContainingPackageLookupFunction}): For
   * each given file path, we look for the nearest ancestor directory (starting with its parent
   * directory), if any, that has a package. The {@link PackageLookupValue} for this package tells
   * us the package root that we should use for the {@link RootedPath} for the {@link
   * FileStateValue} key.
   *
   * <p>For the reverse graph traversal, we are looking for all packages that are transitively
   * reverse dependencies of those {@link FileStateValue} keys. This function returns a collection
   * of SkyKeys whose transitive reverse dependencies must contain the exact same set of packages.
   *
   * <p>Note that there may not be nodes in the graph corresponding to the returned SkyKeys.
   */
  private static Collection<SkyKey> getSkyKeysForFileFragments(
      WalkableGraph graph, Iterable<PathFragment> pathFragments) throws InterruptedException {
    Set<SkyKey> result = new HashSet<>();
    Multimap<PathFragment, PathFragment> currentToOriginal = ArrayListMultimap.create();
    for (PathFragment pathFragment : pathFragments) {
      currentToOriginal.put(pathFragment, pathFragment);
    }
    while (!currentToOriginal.isEmpty()) {
      Multimap<SkyKey, PathFragment> packageLookupKeysToOriginal = ArrayListMultimap.create();
      Multimap<SkyKey, PathFragment> packageLookupKeysToCurrent = ArrayListMultimap.create();
      for (Map.Entry<PathFragment, PathFragment> entry : currentToOriginal.entries()) {
        PathFragment current = entry.getKey();
        PathFragment original = entry.getValue();
        for (SkyKey packageLookupKey : getPkgLookupKeysForFile(current)) {
          packageLookupKeysToOriginal.put(packageLookupKey, original);
          packageLookupKeysToCurrent.put(packageLookupKey, current);
        }

        // The WORKSPACE file is a transitive dependency of every package. Unfortunately, there is
        // no specific SkyValue that we can use to figure out under which package path entries it
        // lives so we add a dependency on the WorkspaceNameValue key.
        if (original.equals(current) && WorkspaceFileHelper.matchWorkspaceFileName(original)) {
          // TODO(mschaller): this should not be checked at runtime. These are constants!
          Preconditions.checkState(
              LabelConstants.WORKSPACE_FILE_NAME
                  .getParentDirectory()
                  .equals(PathFragment.EMPTY_FRAGMENT),
              LabelConstants.WORKSPACE_FILE_NAME);
          result.add(WorkspaceNameValue.key());
        }
      }
      Map<SkyKey, SkyValue> lookupValues =
          graph.getSuccessfulValues(packageLookupKeysToOriginal.keySet());
      for (Map.Entry<SkyKey, SkyValue> entry : lookupValues.entrySet()) {
        SkyKey packageLookupKey = entry.getKey();
        PackageLookupValue packageLookupValue = (PackageLookupValue) entry.getValue();
        if (packageLookupValue.packageExists()) {
          Collection<PathFragment> originalFiles =
              packageLookupKeysToOriginal.get(packageLookupKey);
          Preconditions.checkState(!originalFiles.isEmpty(), entry);
          for (PathFragment fileName : originalFiles) {
            result.add(
                FileStateValue.key(
                    RootedPath.toRootedPath(packageLookupValue.getRoot(), fileName)));
          }
          for (PathFragment current : packageLookupKeysToCurrent.get(packageLookupKey)) {
            currentToOriginal.removeAll(current);
          }
        }
      }
      Multimap<PathFragment, PathFragment> newCurrentToOriginal = ArrayListMultimap.create();
      for (PathFragment pathFragment : currentToOriginal.keySet()) {
        PathFragment parent = pathFragment.getParentDirectory();
        if (parent != null) {
          newCurrentToOriginal.putAll(parent, currentToOriginal.get(pathFragment));
        }
      }
      currentToOriginal = newCurrentToOriginal;
    }
    return result;
  }

  /**
   * Returns package lookup keys for looking up the package root for which there may be a relevant
   * {@link FileStateValue} node in the graph for {@code originalFileFragment}, which is assumed to
   * be a file path.
   *
   * <p>This is a helper function for {@link #getSkyKeysForFileFragments}.
   */
  private static Iterable<SkyKey> getPkgLookupKeysForFile(PathFragment currentPathFragment) {
    PathFragment parentPathFragment = currentPathFragment.getParentDirectory();
    return parentPathFragment == null
        ? ImmutableList.of()
        : ImmutableList.of(
            PackageLookupValue.key(PackageIdentifier.createInMainRepo(parentPathFragment)));
  }
}
