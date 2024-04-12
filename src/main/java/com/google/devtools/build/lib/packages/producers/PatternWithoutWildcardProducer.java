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
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.packages.producers.GlobComputationProducer.GlobDetail;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.StateMachine;
import java.util.Set;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * {@link PatternWithoutWildcardProducer} is a sub-{@link StateMachine} created by {@link
 * FragmentProducer}. It handles glob pattern fragment which does not contain any wildcard
 * characters ({@code *} or {@code **}).
 *
 * <p>When the pattern does not contain any wildcard character, the path is uniquely determined. So
 * it is only necessary to query the {@link FileValue} ending with this glob pattern fragment. If a
 * such file exists, we handle it by creating the {@link DirectoryDirentProducer} under this {@link
 * #filePath}.
 */
final class PatternWithoutWildcardProducer implements StateMachine, Consumer<SkyValue> {

  // -------------------- Input --------------------
  private final GlobDetail globDetail;

  /** The {@link PathFragment} of the file containing the package fragments. */
  private final PathFragment filePath;

  private final int fragmentIndex;

  // -------------------- Internal State --------------------
  private FileValue fileValue = null;
  @Nullable private final Set<Pair<PathFragment, Integer>> visitedGlobSubTasks;

  // -------------------- Output --------------------
  private final FragmentProducer.ResultSink resultSink;

  PatternWithoutWildcardProducer(
      GlobDetail globDetail,
      PathFragment filePath,
      int fragmentIndex,
      FragmentProducer.ResultSink resultSink,
      @Nullable Set<Pair<PathFragment, Integer>> visitedGlobSubTasks) {
    this.globDetail = globDetail;
    this.filePath = filePath;
    this.fragmentIndex = fragmentIndex;
    this.resultSink = resultSink;
    this.visitedGlobSubTasks = visitedGlobSubTasks;
  }

  @Override
  public StateMachine step(Tasks tasks) {
    tasks.lookUp(
        FileValue.key(RootedPath.toRootedPath(globDetail.packageRoot(), filePath)),
        (Consumer<SkyValue>) this);
    return this::processFileValue;
  }

  @Override
  public void accept(SkyValue skyValue) {
    fileValue = (FileValue) skyValue;
  }

  /** Processes {@link FileValue} for the input {@link #filePath}. */
  private StateMachine processFileValue(Tasks tasks) {
    Preconditions.checkNotNull(fileValue);
    if (!fileValue.exists()) {
      // Early exit if fileValue is null due to exception thrown during computation or the file does
      // not exist.
      return DONE;
    }

    if (fileValue.isDirectory()) {
      return new DirectoryDirentProducer(
          globDetail, filePath, fragmentIndex, resultSink, visitedGlobSubTasks);
    }
    FragmentProducer.maybeAddFileMatchingToResult(filePath, fragmentIndex, globDetail, resultSink);
    return DONE;
  }
}
