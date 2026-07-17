// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages.producers;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.Globber.Operation;
import com.google.devtools.build.lib.packages.producers.GlobComputationProducer.GlobDetail;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.state.StateMachine;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Recursively created to start handling each pattern fragment. Based on whether wildcard character
 * exists in the pattern, it creates either {@link PatternWithoutWildcardProducer} or {@link
 * PatternWithWildcardProducer} producer.
 *
 * <p>{@link FragmentProducer} also handles special condition when the pattern is {@code **} by
 * immediately skipping the {@code **} and creating the next {@link FragmentProducer}.
 */
final class FragmentProducer implements StateMachine {

  /** Accepts matching {@link PathFragment}s or any exceptions wrapped in {@link GlobError}. */
  interface ResultSink {

    void acceptPathFragmentWithPackageFragment(PathFragment pathFragment);

    void acceptGlobError(GlobError error);
  }

  // -------------------- Input --------------------
  private final GlobDetail globDetail;

  /**
   * Contains package fragments of the {@link PackageIdentifier}. It is guaranteed that:
   *
   * <ul>
   *   <li>{@link #base} is a directory;
   *   <li>there is no subpackage under {@link #base}, when {@link #base} is not the package
   *       fragment.
   * </ul>
   */
  private final PathFragment base;

  /** Position of the pattern in {@link GlobDetail#patternFragments()} to be processed. */
  private final int fragmentIndex;

  /**
   * The visited set is created to prevent potential duplicate work when handling glob pattern
   * containing multiple {@code **}s.
   *
   * <p>Each pair in the {@link #visitedGlobSubTasks} reflects that some previous {@link
   * FragmentProducer} has already processed a state when the {@link #base} is at the {@code
   * pair.getFirst()} location and {@link #fragmentIndex} at {@code pair.getSecond()} position in
   * the {@link GlobDetail#patternFragments()}.
   *
   * <p>Consider this concrete example: {@code glob(['**\/a/**\/foo.txt'])} with the only file being
   * {@code a/a/foo.txt}.
   *
   * <p>There are multiple routes to reach a point when a {@code FragmentProducer} whose base is
   * {@code a/a/foo.txt} and fragmentIndex is 3 (at "foo.txt") should be created.
   *
   * <ul>
   *   <li>One route starts by recursively globbing 'a/**\/foo.txt' in the base directory of the
   *       package.
   *   <li>Another route starts by recursively globbing '**\/a/**\/foo.txt' in subdirectory 'a'.
   * </ul>
   *
   * <p>{@link #visitedGlobSubTasks} prevents such a {@code FragmentProducer} from being created and
   * processed for the second time, and thus reduces duplicate computation.
   */
  @Nullable private final Set<Pair<PathFragment, Integer>> visitedGlobSubTasks;

  // -------------------- Output --------------------
  final ResultSink resultSink;

  FragmentProducer(
      GlobDetail globDetail,
      PathFragment base,
      int fragmentIndex,
      @Nullable Set<Pair<PathFragment, Integer>> visitedGlobSubTasks,
      ResultSink resultSink) {
    // Make sure condition (1) glob patterns contains multiple `**`s and condition (2)
    // `visitedGlobSubTasks` is null should be the same.
    Preconditions.checkState(
        globDetail.containsMultipleDoubleStars() == (visitedGlobSubTasks != null));
    this.globDetail = globDetail;
    this.base = base;
    this.fragmentIndex = fragmentIndex;
    this.visitedGlobSubTasks = visitedGlobSubTasks;
    this.resultSink = resultSink;
  }

  @Override
  public StateMachine step(Tasks tasks) {
    Preconditions.checkState(fragmentIndex < globDetail.patternFragments().size());

    String patternFragment = globDetail.patternFragments().get(fragmentIndex);
    if (!patternFragment.equals("**")) {
      return handlePatternFragment(patternFragment);
    }

    // It is valid that `**` matches nothing, which is handled in the if-else block below.
    if (fragmentIndex < globDetail.patternFragments().size() - 1) {
      // In the case when `**` is not the last pattern, skip `**` and directly move onto the next
      // pattern fragment.
      if (visitedGlobSubTasks == null
          || visitedGlobSubTasks.add(Pair.of(base, fragmentIndex + 1))) {
        tasks.enqueue(
            new FragmentProducer(
                globDetail, base, fragmentIndex + 1, visitedGlobSubTasks, resultSink));
      }
    } else {
      // In the case when `**` is the last pattern, add `base` to result when operator is
      // FILES_AND_DIRS.
      if (globDetail.globOperation().equals(Operation.FILES_AND_DIRS)
          && !base.equals(globDetail.packageIdentifier().getPackageFragment())) {
        resultSink.acceptPathFragmentWithPackageFragment(base);
      }
    }

    // Handle the case when `**` does not match an empty fragment.
    return handlePatternFragment(patternFragment);
  }

  private StateMachine handlePatternFragment(String patternFragment) {
    if (!patternFragment.contains("*") && !patternFragment.contains("?")) {
      return new PatternWithoutWildcardProducer(
          globDetail,
          base.getChild(patternFragment),
          fragmentIndex,
          resultSink,
          visitedGlobSubTasks);
    }
    return new PatternWithWildcardProducer(
        globDetail, base, fragmentIndex, resultSink, visitedGlobSubTasks);
  }

  /** Returns if a matching path at the given pattern index should be added to the result. */
  static boolean shouldAddFileMatchingToResult(int fragmentIndex, GlobDetail globDetail) {
    if (globDetail.globOperation().equals(Operation.SUBPACKAGES)) {
      return false;
    }
    if (fragmentIndex < globDetail.patternFragments().size() - 1) {
      return false;
    }
    return true;
  }
}
