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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.sun.management.GarbageCollectionNotificationInfo;
import com.sun.management.GcInfo;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.MemoryUsage;
import java.util.List;
import javax.management.ListenerNotFoundException;
import javax.management.Notification;
import javax.management.NotificationEmitter;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RetainedHeapLimiter}. */
@RunWith(JUnit4.class)
public final class RetainedHeapLimiterTest {

  private interface NotificationBean extends GarbageCollectorMXBean, NotificationEmitter {}

  private final NotificationBean mockBean = mock(NotificationBean.class);
  private final GarbageCollectorMXBean mockUselessBean = mock(GarbageCollectorMXBean.class);

  @Before
  public void initMocks() {
    when(mockBean.getMemoryPoolNames()).thenReturn(new String[] {"not tenured", "CMS Old Gen"});
    when(mockUselessBean.getMemoryPoolNames()).thenReturn(new String[] {"assistant", "adjunct"});
  }

  @Test
  public void findBeans() {
    assertThat(
            RetainedHeapLimiter.findTenuredCollectorBeans(
                ImmutableList.of(mockUselessBean, mockBean)))
        .containsExactly(mockBean);
  }

  @Test
  public void smoke() throws AbruptExitException, ListenerNotFoundException {
    RetainedHeapLimiter underTest =
        RetainedHeapLimiter.createFromBeans(
            ImmutableList.of(mockUselessBean, mockBean), BugReporter.defaultInstance());

    underTest.updateThreshold(100);
    verify(mockBean, never()).addNotificationListener(underTest, null, null);
    verify(mockBean, never()).removeNotificationListener(underTest, null, null);

    underTest.updateThreshold(90);
    verify(mockBean).addNotificationListener(underTest, null, null);
    verify(mockBean, never()).removeNotificationListener(underTest, null, null);

    underTest.updateThreshold(80);
    // No additional calls.
    verify(mockBean).addNotificationListener(underTest, null, null);
    verify(mockBean, never()).removeNotificationListener(underTest, null, null);

    underTest.updateThreshold(100);
    verify(mockBean).addNotificationListener(underTest, null, null);
    verify(mockBean).removeNotificationListener(underTest, null, null);
  }

  @Test
  public void noTenuredSpaceFound() throws AbruptExitException {
    RetainedHeapLimiter underTest =
        RetainedHeapLimiter.createFromBeans(
            ImmutableList.of(mockUselessBean), BugReporter.defaultInstance());
    verify(mockUselessBean, times(2)).getMemoryPoolNames();

    underTest.updateThreshold(100);
    verifyNoMoreInteractions(mockUselessBean);

    AbruptExitException e =
        assertThrows(AbruptExitException.class, () -> underTest.updateThreshold(80));
    FailureDetails.FailureDetail failureDetail = e.getDetailedExitCode().getFailureDetail();
    assertThat(failureDetail.getMessage())
        .contains("unable to watch for GC events to exit JVM when 80% of heap is used");
    assertThat(failureDetail.getMemoryOptions().getCode())
        .isEqualTo(
            FailureDetails.MemoryOptions.Code
                .EXPERIMENTAL_OOM_MORE_EAGERLY_NO_TENURED_COLLECTORS_FOUND);
  }

  @Test
  public void underThreshold_noOom() throws Exception {
    RetainedHeapLimiter underTest =
        RetainedHeapLimiter.createFromBeans(
            ImmutableList.of(mockBean), BugReporter.defaultInstance());
    underTest.updateThreshold(90);

    underTest.handleNotification(percentUsedAfterForcedGc(89), null);
  }

  @Test
  public void overThreshold_oom() throws Exception {
    class OomThrowingBugReporter implements BugReporter {
      @Override
      public void sendBugReport(Throwable exception) {
        throw new IllegalStateException(exception);
      }

      @Override
      public void sendBugReport(Throwable exception, List<String> args, String... values) {
        throw new IllegalStateException(exception);
      }

      @Override
      public RuntimeException handleCrash(Throwable exception, String... args) {
        assertThat(exception).isInstanceOf(OutOfMemoryError.class);
        throw (OutOfMemoryError) exception;
      }
    }
    RetainedHeapLimiter underTest =
        RetainedHeapLimiter.createFromBeans(
            ImmutableList.of(mockBean), new OomThrowingBugReporter());
    underTest.updateThreshold(90);

    OutOfMemoryError oom =
        assertThrows(
            OutOfMemoryError.class,
            () -> underTest.handleNotification(percentUsedAfterForcedGc(91), null));

    assertThat(oom).hasMessageThat().contains("forcing exit due to GC thrashing");
    assertThat(oom).hasMessageThat().contains("tenured space is more than 90% occupied");
  }

  private static Notification percentUsedAfterForcedGc(int percentUsed) {
    checkArgument(percentUsed >= 0 && percentUsed <= 100, percentUsed);
    MemoryUsage memoryUsage = mock(MemoryUsage.class);
    when(memoryUsage.getUsed()).thenReturn((long) percentUsed);
    when(memoryUsage.getMax()).thenReturn(100L);

    GcInfo gcInfo = mock(GcInfo.class);
    when(gcInfo.getMemoryUsageAfterGc()).thenReturn(ImmutableMap.of("CMS Old Gen", memoryUsage));

    GarbageCollectionNotificationInfo notificationInfo =
        new GarbageCollectionNotificationInfo("name", "action", "System.gc()", gcInfo);

    Notification notification =
        new Notification(
            GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION, "test", 123);
    notification.setUserData(notificationInfo.toCompositeData(null));
    return notification;
  }
}
