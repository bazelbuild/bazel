// Copyright 2021 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.events.NullEventHandler;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link EvaluationContext}. */
@RunWith(JUnit4.class)
public class EvaluationContextTest {
  @Test
  public void testGetCPUHeavySkyKeysThreadPoolSize_executionPhase_returnNegativeOne() {
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setNumThreads(1)
            .setEventHandler(NullEventHandler.INSTANCE)
            .setExecutionPhase()
            .setCPUHeavySkyKeysThreadPoolSize(42)
            .build();

    assertThat(evaluationContext.getCPUHeavySkyKeysThreadPoolSize()).isEqualTo(-1);
  }

  @Test
  public void testGetCPUHeavySkyKeysThreadPoolSize_nonExecutionPhase_returnSpecifiedPoolSize() {
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setNumThreads(1)
            .setEventHandler(NullEventHandler.INSTANCE)
            .setCPUHeavySkyKeysThreadPoolSize(42)
            .build();

    assertThat(evaluationContext.getCPUHeavySkyKeysThreadPoolSize()).isEqualTo(42);
  }
}
