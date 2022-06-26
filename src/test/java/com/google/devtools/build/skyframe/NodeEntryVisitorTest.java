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

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

import com.google.devtools.build.lib.concurrent.MultiThreadPoolsQuiescingExecutor;
import com.google.devtools.build.lib.concurrent.MultiThreadPoolsQuiescingExecutor.ThreadPoolType;
import com.google.devtools.build.skyframe.ParallelEvaluatorContext.RunnableMaker;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link NodeEntryVisitor}. */
@RunWith(JUnit4.class)
public class NodeEntryVisitorTest {
  @Test
  public void enqueueEvaluation_multiThreadPoolsQuiescingExecutor_nonCPUHeavyKey() {
    MultiThreadPoolsQuiescingExecutor executor = mock(MultiThreadPoolsQuiescingExecutor.class);
    NodeEntryVisitor nodeEntryVisitor =
        new NodeEntryVisitor(
            executor, mock(DirtyTrackingProgressReceiver.class), mock(RunnableMaker.class));
    SkyKey nonCPUHeavyKey = mock(SkyKey.class);

    nodeEntryVisitor.enqueueEvaluation(nonCPUHeavyKey, Integer.MAX_VALUE);

    verify(executor).execute(any(), eq(ThreadPoolType.REGULAR));
  }

  @Test
  public void enqueueEvaluation_multiThreadPoolsQuiescingExecutor_cpuHeavyKey() {
    MultiThreadPoolsQuiescingExecutor executor = mock(MultiThreadPoolsQuiescingExecutor.class);
    NodeEntryVisitor nodeEntryVisitor =
        new NodeEntryVisitor(
            executor, mock(DirtyTrackingProgressReceiver.class), mock(RunnableMaker.class));
    CPUHeavySkyKey cpuHeavyKey = mock(CPUHeavySkyKey.class);

    nodeEntryVisitor.enqueueEvaluation(cpuHeavyKey, Integer.MAX_VALUE);

    verify(executor).execute(any(), eq(ThreadPoolType.CPU_HEAVY));
  }
}
