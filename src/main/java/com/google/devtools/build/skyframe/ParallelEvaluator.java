// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.skyframe;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.skyframe.MemoizingEvaluator.EmittedEventState;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * An {@link AbstractExceptionalParallelEvaluator} implementing {@link Evaluator}, for when the only
 * checked exception throwable by evaluation is {@link InterruptedException}.
 *
 * <p>This class is not intended for direct use, and is only exposed as public for use in evaluation
 * implementations outside of this package.
 */
public class ParallelEvaluator extends AbstractExceptionalParallelEvaluator<RuntimeException>
    implements Evaluator {

  public ParallelEvaluator(
      ProcessableGraph graph,
      Version graphVersion,
      ImmutableMap<SkyFunctionName, ? extends SkyFunction> skyFunctions,
      final ExtendedEventHandler reporter,
      EmittedEventState emittedEventState,
      EventFilter storedEventFilter,
      ErrorInfoManager errorInfoManager,
      boolean keepGoing,
      DirtyTrackingProgressReceiver progressReceiver,
      GraphInconsistencyReceiver graphInconsistencyReceiver,
      Supplier<ExecutorService> executorService,
      CycleDetector cycleDetector,
      EvaluationVersionBehavior evaluationVersionBehavior) {
    super(
        graph,
        graphVersion,
        skyFunctions,
        reporter,
        emittedEventState,
        storedEventFilter,
        errorInfoManager,
        keepGoing,
        progressReceiver,
        graphInconsistencyReceiver,
        executorService,
        cycleDetector,
        evaluationVersionBehavior);
  }

  @Override
  @ThreadCompatible
  public <T extends SkyValue> EvaluationResult<T> eval(Iterable<? extends SkyKey> skyKeys)
      throws InterruptedException {
    return this.evalExceptionally(skyKeys);
  }

  @Override
  Map<SkyKey, ValueWithMetadata> bubbleErrorUpExceptionally(
      ErrorInfo leafFailure, SkyKey errorKey, Iterable<SkyKey> roots, Set<SkyKey> rdepsToBubbleUpTo)
      throws InterruptedException {
    return super.bubbleErrorUp(leafFailure, errorKey, roots, rdepsToBubbleUpTo);
  }

  @Override
  <T extends SkyValue> EvaluationResult<T> constructResultExceptionally(
      Iterable<SkyKey> skyKeys,
      @Nullable Map<SkyKey, ValueWithMetadata> bubbleErrorInfo,
      boolean catastrophe)
      throws InterruptedException {
    return super.constructResult(skyKeys, bubbleErrorInfo, catastrophe);
  }
}
