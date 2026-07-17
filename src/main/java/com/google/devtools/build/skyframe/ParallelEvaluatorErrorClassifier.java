// Copyright 2022 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.concurrent.ErrorClassifier;

/** An {@link ErrorClassifier} implementation for {@link ParallelEvaluator}. */
public final class ParallelEvaluatorErrorClassifier extends ErrorClassifier {
  private static final ParallelEvaluatorErrorClassifier INSTANCE =
      new ParallelEvaluatorErrorClassifier();

  public static ParallelEvaluatorErrorClassifier instance() {
    return INSTANCE;
  }

  private ParallelEvaluatorErrorClassifier() {}

  @Override
  protected ErrorClassification classifyException(Exception e) {
    if (e instanceof SchedulerException) {
      return ErrorClassification.CRITICAL;
    }
    if (e instanceof RuntimeException) {
      // We treat non-SchedulerException RuntimeExceptions as more severe than
      // SchedulerExceptions so that AbstractQueueVisitor will propagate instances of the
      // former. They indicate actual Blaze bugs, rather than normal Skyframe evaluation
      // control flow.
      return ErrorClassification.CRITICAL_AND_LOG;
    }
    return ErrorClassification.NOT_CRITICAL;
  }
}
