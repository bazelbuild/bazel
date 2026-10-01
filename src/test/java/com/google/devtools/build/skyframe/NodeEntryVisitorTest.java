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
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.github.benmanes.caffeine.cache.Cache;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.concurrent.MultiThreadPoolsQuiescingExecutor;
import com.google.devtools.build.lib.concurrent.MultiThreadPoolsQuiescingExecutor.ThreadPoolType;
import com.google.devtools.build.lib.concurrent.QuiescingExecutor;
import com.google.devtools.build.lib.concurrent.QuiescingTask;
import com.google.devtools.build.skyframe.ParallelEvaluatorContext.RunnableMaker;
import com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

/** Tests for {@link NodeEntryVisitor}. */
@RunWith(JUnit4.class)
public class NodeEntryVisitorTest {
  @Rule public final MockitoRule mockito = MockitoJUnit.rule();

  @Mock private MultiThreadPoolsQuiescingExecutor executor;
  @Mock private InflightTrackingProgressReceiver receiver;
  @Mock private RunnableMaker runnableMaker;
  @Mock private Cache<SkyKey, SkyKeyComputeState> stateCache;

  private static QuiescingTask createTestTask() {
    return new QuiescingTask(mock(AbstractQueueVisitor.class)) {
      @Override
      public void runCore() {}
    };
  }

  @Test
  public void enqueueEvaluation_multiThreadPoolsQuiescingExecutor_nonCPUHeavyKey() {
    NodeEntryVisitor nodeEntryVisitor =
        new NodeEntryVisitor(executor, receiver, runnableMaker, stateCache);
    SkyKey nonCpuHeavyKey = mock(SkyKey.class);

    nodeEntryVisitor.enqueueEvaluation(nonCpuHeavyKey, null);

    verify(executor).execute((Runnable) any(), eq(ThreadPoolType.REGULAR), anyBoolean());
  }

  @Test
  public void enqueueEvaluation_multiThreadPoolsQuiescingExecutor_cpuHeavyKey() {
    NodeEntryVisitor nodeEntryVisitor =
        new NodeEntryVisitor(executor, receiver, runnableMaker, stateCache);
    CPUHeavySkyKey cpuHeavyKey = mock(CPUHeavySkyKey.class);

    nodeEntryVisitor.enqueueEvaluation(cpuHeavyKey, null);

    verify(executor).execute((Runnable) any(), eq(ThreadPoolType.CPU_HEAVY), anyBoolean());
  }

  @Test
  public void enqueueEvaluation_multiThreadPoolsQuiescingExecutor_quiescingTask_nonCpuHeavyKey() {
    QuiescingTask task = createTestTask();
    when(runnableMaker.make(any())).thenReturn(task);
    NodeEntryVisitor nodeEntryVisitor =
        new NodeEntryVisitor(executor, receiver, runnableMaker, stateCache);
    SkyKey nonCpuHeavyKey = mock(SkyKey.class);

    nodeEntryVisitor.enqueueEvaluation(nonCpuHeavyKey, null);

    verify(executor).execute(eq(task), eq(ThreadPoolType.REGULAR), anyBoolean());
  }

  @Test
  public void enqueueEvaluation_multiThreadPoolsQuiescingExecutor_quiescingTask_cpuHeavyKey() {
    QuiescingTask task = createTestTask();
    when(runnableMaker.make(any())).thenReturn(task);
    NodeEntryVisitor nodeEntryVisitor =
        new NodeEntryVisitor(executor, receiver, runnableMaker, stateCache);
    CPUHeavySkyKey cpuHeavyKey = mock(CPUHeavySkyKey.class);

    nodeEntryVisitor.enqueueEvaluation(cpuHeavyKey, null);

    verify(executor).execute(eq(task), eq(ThreadPoolType.CPU_HEAVY), anyBoolean());
  }

  @Test
  public void enqueueEvaluation_regularQuiescingExecutor_quiescingTask() {
    QuiescingExecutor regularExecutor = mock(QuiescingExecutor.class);
    QuiescingTask task = createTestTask();
    when(runnableMaker.make(any())).thenReturn(task);
    NodeEntryVisitor nodeEntryVisitor =
        new NodeEntryVisitor(regularExecutor, receiver, runnableMaker, stateCache);
    SkyKey key = mock(SkyKey.class);

    nodeEntryVisitor.enqueueEvaluation(key, null);

    verify(regularExecutor).execute(task);
  }
}
