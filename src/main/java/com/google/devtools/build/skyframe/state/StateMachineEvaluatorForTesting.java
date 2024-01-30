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
package com.google.devtools.build.skyframe.state;

import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.ArrayList;
import java.util.HashMap;
import javax.annotation.Nullable;

/**
 * Evaluates {@link StateMachine} using a given {@link MemoizingEvaluator} for testing.
 *
 * <p>As the {@link StateMachine} requests dependencies, delegates requests to the underlying graph
 * and records missing values. Then evaluates any missing dependencies before resuming the {@link
 * StateMachine}.
 *
 * <p>Only supports {@code keepGoing} evaluations.
 */
public final class StateMachineEvaluatorForTesting {
  private final MemoizingEvaluator evaluator;
  private final Driver driver;

  /** Values are either {@link SkyValue} or {@link Exception}. */
  private final HashMap<SkyKey, Object> previousResults = new HashMap<>();

  /**
   * Runs the given {@link StateMachine}.
   *
   * @return the result of the last evalution, if any, for error handling.
   */
  @Nullable // Null if there were no evaluations.
  public static EvaluationResult<SkyValue> run(
      StateMachine root, MemoizingEvaluator evaluator, EvaluationContext context)
      throws InterruptedException {
    return new StateMachineEvaluatorForTesting(root, evaluator).evaluate(context);
  }

  private StateMachineEvaluatorForTesting(StateMachine root, MemoizingEvaluator evaluator) {
    this.driver = new Driver(root);
    this.evaluator = evaluator;
  }

  private EvaluationResult<SkyValue> evaluate(EvaluationContext context)
      throws InterruptedException {
    var missing = new ArrayList<SkyKey>();
    var env =
        new EnvironmentForUtilities(
            skyKey -> {
              var value = previousResults.get(skyKey);
              if (value != null) {
                return value;
              }
              missing.add(skyKey);
              return null;
            });

    EvaluationResult<SkyValue> result = null;
    boolean hasError = false;
    while (!driver.drive(env)) {
      if (hasError) {
        return result; // Exits if there was an error in the previous round.
      }

      result = evaluator.evaluate(missing, context);
      for (SkyKey key : missing) {
        SkyValue value = result.get(key);
        if (value != null) {
          previousResults.put(key, value);
          continue;
        }
        // Marks an error. The state machine will run one more time for "error bubbling" before
        // exiting.
        hasError = true;
        ErrorInfo error = result.getError(key);
        if (error == null) {
          continue;
        }
        Exception exception = error.getException();
        if (exception != null) {
          previousResults.put(key, exception);
        }
        // Otherwise, there might be a cycle.
      }
      missing.clear();
    }
    return result;
  }
}
