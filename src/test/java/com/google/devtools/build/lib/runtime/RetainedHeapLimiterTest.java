// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.withSettings;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.util.AbruptExitException;
import java.lang.management.GarbageCollectorMXBean;
import javax.management.ListenerNotFoundException;
import javax.management.NotificationEmitter;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/**
 * Basic tests for {@link RetainedHeapLimiter}. More realistic tests are hard because {@link
 * RetainedHeapLimiter} intentionally crashes the JVM.
 */
@RunWith(JUnit4.class)
public class RetainedHeapLimiterTest {
  @Test
  public void findBeans() {
    GarbageCollectorMXBean mockUselessBean = Mockito.mock(GarbageCollectorMXBean.class);
    String[] untenuredPoolNames = {"assistant", "adjunct"};
    Mockito.when(mockUselessBean.getMemoryPoolNames()).thenReturn(untenuredPoolNames);
    GarbageCollectorMXBean mockBean =
        Mockito.mock(
            GarbageCollectorMXBean.class,
            withSettings().extraInterfaces(NotificationEmitter.class));
    String[] poolNames = {"not tenured", "CMS Old Gen"};
    Mockito.when(mockBean.getMemoryPoolNames()).thenReturn(poolNames);
    assertThat(
            RetainedHeapLimiter.findTenuredCollectorBeans(
                ImmutableList.of(mockUselessBean, mockBean)))
        .containsExactly(mockBean);
  }

  @Test
  public void smoke() throws AbruptExitException, ListenerNotFoundException {
    GarbageCollectorMXBean mockUselessBean = Mockito.mock(GarbageCollectorMXBean.class);
    String[] untenuredPoolNames = {"assistant", "adjunct"};
    Mockito.when(mockUselessBean.getMemoryPoolNames()).thenReturn(untenuredPoolNames);
    GarbageCollectorMXBean mockBean =
        Mockito.mock(
            GarbageCollectorMXBean.class,
            withSettings().extraInterfaces(NotificationEmitter.class));
    String[] poolNames = {"not tenured", "CMS Old Gen"};
    Mockito.when(mockBean.getMemoryPoolNames()).thenReturn(poolNames);

    RetainedHeapLimiter underTest =
        new RetainedHeapLimiter(ImmutableList.of(mockUselessBean, mockBean));
    underTest.updateThreshold(100);
    Mockito.verify((NotificationEmitter) mockBean, never())
        .addNotificationListener(underTest, null, null);
    Mockito.verify((NotificationEmitter) mockBean, never())
        .removeNotificationListener(underTest, null, null);

    underTest.updateThreshold(90);
    Mockito.verify((NotificationEmitter) mockBean, times(1))
        .addNotificationListener(underTest, null, null);
    Mockito.verify((NotificationEmitter) mockBean, never())
        .removeNotificationListener(underTest, null, null);

    underTest.updateThreshold(80);
    // No additional calls.
    Mockito.verify((NotificationEmitter) mockBean, times(1))
        .addNotificationListener(underTest, null, null);
    Mockito.verify((NotificationEmitter) mockBean, never())
        .removeNotificationListener(underTest, null, null);

    underTest.updateThreshold(100);
    Mockito.verify((NotificationEmitter) mockBean, times(1))
        .addNotificationListener(underTest, null, null);
    Mockito.verify((NotificationEmitter) mockBean, times(1))
        .removeNotificationListener(underTest, null, null);
  }
}
