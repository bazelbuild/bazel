// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.producers;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.StateMachine;
import com.google.devtools.build.skyframe.state.StateMachineEvaluatorForTesting;

/** Base class for tests of producers. */
public abstract class ProducerTestCase extends BuildViewTestCase {
  @Override
  protected void useConfiguration(String... args) throws Exception {
    // Do nothing, some of the producers under test are used in standard configuration creation.
  }

  /**
   * Use a {@link StateMachineEvaluatorForTesting} to drive the given {@link StateMachine} until it
   * finishes (with a result or an error). Results should be retrieved from whatever result sink the
   * {@link StateMachine} is designed for.
   *
   * @return {@code true} on success
   */
  public boolean executeProducer(StateMachine producer) throws InterruptedException {
    EvaluationContext context =
        EvaluationContext.newBuilder()
            .setKeepGoing(true)
            .setParallelism(SkyframeExecutor.DEFAULT_THREAD_COUNT)
            .setEventHandler(reporter)
            .build();
    try {
      getSkyframeExecutor().getSkyframeBuildView().enableAnalysis(true);
      EvaluationResult<SkyValue> result =
          StateMachineEvaluatorForTesting.run(
              producer, getSkyframeExecutor().getEvaluator(), context);
      if (result != null) {
        return !result.hasError();
      }
    } finally {
      getSkyframeExecutor().getSkyframeBuildView().enableAnalysis(false);
    }
    return true;
  }
}
