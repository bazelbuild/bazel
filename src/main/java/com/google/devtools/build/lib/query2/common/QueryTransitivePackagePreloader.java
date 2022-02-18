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
package com.google.devtools.build.lib.query2.common;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.skyframe.DetailedException;
import com.google.devtools.build.lib.skyframe.TransitiveTargetKey;
import com.google.devtools.build.lib.skyframe.TransitiveTargetValue;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;
import javax.annotation.Nullable;

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

  /**
   * Unless every top-level key in error depends on a cycle, throws a {@link QueryException}
   * (derived from an error in {@code result}).
   */
  public static void maybeThrowQueryExceptionForResultWithError(
      EvaluationResult<SkyValue> result,
      Iterable<? extends SkyKey> roots,
      QueryExpression caller,
      String operation)
      throws QueryException {
    Exception exception = result.getCatastrophe();
    if (exception != null) {
      throw throwException(exception, caller, operation, result);
    }

    // Catastrophe not present: look at top-level keys now.
    boolean foundCycle = false;
    for (ErrorInfo errorInfo : result.errorMap().values()) {
      if (!errorInfo.getCycleInfo().isEmpty()) {
        foundCycle = true;
      } else {
        exception = errorInfo.getException();
        if (exception instanceof DetailedException) {
          break;
        }
      }
    }

    if (exception != null) {
      throw throwException(exception, caller, operation, result);
    }
    Preconditions.checkState(
        foundCycle, "No cycle or exception found in result with error: %s %s", result, roots);
  }

  private static QueryException throwException(
      Exception exception,
      QueryExpression caller,
      String operation,
      EvaluationResult<SkyValue> resultForDebugging)
      throws QueryException {
    FailureDetails.FailureDetail failureDetail;
    if (!(exception instanceof DetailedException)) {
      BugReport.sendBugReport(
          new IllegalStateException(
              "Non-detailed exception found for " + operation + ": " + resultForDebugging,
              exception));
      failureDetail =
          FailureDetails.FailureDetail.newBuilder()
              .setQuery(
                  FailureDetails.Query.newBuilder()
                      .setCode(FailureDetails.Query.Code.NON_DETAILED_ERROR))
              .build();
    } else {
      failureDetail = ((DetailedException) exception).getDetailedExitCode().getFailureDetail();
    }
    throw new QueryException(
        caller, operation + " failed: " + exception.getMessage(), exception, failureDetail);
  }

  /** Loads the specified {@link TransitiveTargetValue}s. */
  public void preloadTransitiveTargets(
      ExtendedEventHandler eventHandler,
      Iterable<Label> labelsToVisit,
      boolean keepGoing,
      int parallelThreads,
      @Nullable QueryExpression callerForError)
      throws InterruptedException, QueryException {
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
    EvaluationResult<SkyValue> result =
        memoizingEvaluatorSupplier.get().evaluate(valueNames, evaluationContext);
    if (callerForError != null && result.hasError()) {
      maybeThrowQueryExceptionForResultWithError(
          result, labelsToVisit, callerForError, "preloading transitive closure");
    }
  }
}
