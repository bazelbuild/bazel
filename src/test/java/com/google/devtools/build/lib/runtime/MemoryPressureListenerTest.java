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

package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.sun.management.GarbageCollectionNotificationInfo;
import com.sun.management.GcInfo;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.MemoryUsage;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import javax.management.Notification;
import javax.management.NotificationEmitter;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link MemoryPressureListener}. */
@RunWith(JUnit4.class)
public final class MemoryPressureListenerTest {
  private interface NotificationBean extends GarbageCollectorMXBean, NotificationEmitter {}

  private static final String TENURED_SPACE_NAME = "CMS Old Gen";
  private final NotificationBean mockBean = mock(NotificationBean.class);
  private final NotificationBean mockUselessBean = mock(NotificationBean.class);
  private final EventBus eventBus = new EventBus();
  private final List<MemoryPressureEvent> events = new ArrayList<>();

  @Before
  public void initMocks() {
    when(mockBean.getMemoryPoolNames())
        .thenReturn(new String[] {"not tenured", TENURED_SPACE_NAME});
    when(mockUselessBean.getMemoryPoolNames()).thenReturn(new String[] {"assistant", "adjunct"});
  }

  @Before
  public void registerSubscriber() {
    eventBus.register(
        new Object() {
          @Subscribe
          public void handle(MemoryPressureEvent event) {
            events.add(event);
          }
        });
  }

  @Test
  public void findBeans() {
    assertThat(
            MemoryPressureListener.findTenuredCollectorBeans(
                ImmutableList.of(mockUselessBean, mockBean)))
        .containsExactly(mockBean);
  }

  @Test
  public void createFromBeans_throwsIfNoTenuredSpaceBean() {
    assertThrows(
        IllegalStateException.class,
        () ->
            MemoryPressureListener.createFromBeans(
                ImmutableList.of(mockUselessBean), directExecutor()));
  }

  @Test
  public void simple() {
    MemoryPressureListener underTest =
        MemoryPressureListener.createFromBeans(
            ImmutableList.of(mockUselessBean, mockBean), directExecutor());
    underTest.initForInvocation(
        eventBus, mock(GcThrashingDetector.class), mock(GcChurningDetector.class));
    verify(mockBean).addNotificationListener(underTest, null, null);
    verify(mockUselessBean, never()).addNotificationListener(any(), any(), any());

    GcInfo mockGcInfo = mock(GcInfo.class);
    String nonTenuredSpaceName = "nope";
    MemoryUsage mockMemoryUsageForNonTenuredSpace = mock(MemoryUsage.class);
    MemoryUsage mockMemoryUsageForTenuredSpace = mock(MemoryUsage.class);
    when(mockMemoryUsageForTenuredSpace.getUsed()).thenReturn(42L);
    when(mockMemoryUsageForTenuredSpace.getMax()).thenReturn(100L);
    when(mockGcInfo.getMemoryUsageAfterGc())
        .thenReturn(
            ImmutableMap.of(
                nonTenuredSpaceName,
                mockMemoryUsageForNonTenuredSpace,
                TENURED_SPACE_NAME,
                mockMemoryUsageForTenuredSpace));
    when(mockGcInfo.getDuration()).thenReturn(42_000L);

    Notification notification =
        new Notification(
            GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION, "test", 123);
    notification.setUserData(
        new GarbageCollectionNotificationInfo("gcName", "gcAction", "non-manual", mockGcInfo)
            .toCompositeData(null));
    underTest.handleNotification(notification, null);

    assertThat(events)
        .containsExactly(
            MemoryPressureEvent.newBuilder()
                .setWasManualGc(false)
                .setTenuredSpaceUsedBytes(42L)
                .setTenuredSpaceMaxBytes(100L)
                .setDuration(Duration.ofSeconds(42))
                .build());
  }

  @Test
  public void nullEventBus_doNotPublishEvent() {
    MemoryPressureListener underTest =
        MemoryPressureListener.createFromBeans(
            ImmutableList.of(mockUselessBean, mockBean), directExecutor());
    verify(mockBean).addNotificationListener(underTest, null, null);
    verify(mockUselessBean, never()).addNotificationListener(any(), any(), any());

    GcInfo mockGcInfo = mock(GcInfo.class);
    String nonTenuredSpaceName = "nope";
    MemoryUsage mockMemoryUsageForNonTenuredSpace = mock(MemoryUsage.class);
    MemoryUsage mockMemoryUsageForTenuredSpace = mock(MemoryUsage.class);
    when(mockMemoryUsageForTenuredSpace.getUsed()).thenReturn(42L);
    when(mockMemoryUsageForTenuredSpace.getMax()).thenReturn(100L);
    when(mockGcInfo.getMemoryUsageAfterGc())
        .thenReturn(
            ImmutableMap.of(
                nonTenuredSpaceName,
                mockMemoryUsageForNonTenuredSpace,
                TENURED_SPACE_NAME,
                mockMemoryUsageForTenuredSpace));

    Notification notification =
        new Notification(
            GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION, "test", 123);
    notification.setUserData(
        new GarbageCollectionNotificationInfo("gcName", "gcAction", "non-manual", mockGcInfo)
            .toCompositeData(null));
    underTest.handleNotification(notification, null);

    assertThat(events).isEmpty();
  }

  @Test
  public void manualGc() {
    MemoryPressureListener underTest =
        MemoryPressureListener.createFromBeans(ImmutableList.of(mockBean), directExecutor());
    underTest.initForInvocation(
        eventBus, mock(GcThrashingDetector.class), mock(GcChurningDetector.class));
    verify(mockBean).addNotificationListener(underTest, null, null);

    GcInfo mockGcInfo = mock(GcInfo.class);
    MemoryUsage mockMemoryUsageForTenuredSpace = mock(MemoryUsage.class);
    when(mockMemoryUsageForTenuredSpace.getUsed()).thenReturn(42L);
    when(mockMemoryUsageForTenuredSpace.getMax()).thenReturn(100L);
    when(mockGcInfo.getMemoryUsageAfterGc())
        .thenReturn(ImmutableMap.of(TENURED_SPACE_NAME, mockMemoryUsageForTenuredSpace));
    when(mockGcInfo.getDuration()).thenReturn(42_000L);

    Notification notification =
        new Notification(
            GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION, "test", 123);
    notification.setUserData(
        new GarbageCollectionNotificationInfo("gcName", "gcAction", "System.gc()", mockGcInfo)
            .toCompositeData(null));
    underTest.handleNotification(notification, null);

    assertThat(events)
        .containsExactly(
            MemoryPressureEvent.newBuilder()
                .setWasManualGc(true)
                .setTenuredSpaceUsedBytes(42L)
                .setTenuredSpaceMaxBytes(100L)
                .setDuration(Duration.ofSeconds(42))
                .build());
  }

  @Test
  public void doesntInvokeHandlerWhenTenuredSpaceMaxSizeIsZero() {
    MemoryPressureListener underTest =
        MemoryPressureListener.createFromBeans(ImmutableList.of(mockBean), directExecutor());
    underTest.initForInvocation(
        eventBus, mock(GcThrashingDetector.class), mock(GcChurningDetector.class));
    verify(mockBean).addNotificationListener(underTest, null, null);

    GcInfo mockGcInfo = mock(GcInfo.class);
    MemoryUsage mockMemoryUsageForTenuredSpace = mock(MemoryUsage.class);
    when(mockMemoryUsageForTenuredSpace.getUsed()).thenReturn(42L);
    when(mockMemoryUsageForTenuredSpace.getMax()).thenReturn(0L);
    when(mockGcInfo.getMemoryUsageAfterGc())
        .thenReturn(ImmutableMap.of(TENURED_SPACE_NAME, mockMemoryUsageForTenuredSpace));

    Notification notification =
        new Notification(
            GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION, "test", 123);
    notification.setUserData(
        new GarbageCollectionNotificationInfo("gcName", "gcAction", "non-manual", mockGcInfo)
            .toCompositeData(null));
    underTest.handleNotification(notification, null);

    assertThat(events).isEmpty();
  }

  @Test
  public void findsTenuredSpaceWithNonZeroMaxSize() {
    NotificationBean anotherMockBean = mock(NotificationBean.class);
    String anotherTenuredSpaceName = "G1 Old Gen";
    when(anotherMockBean.getMemoryPoolNames()).thenReturn(new String[] {anotherTenuredSpaceName});

    MemoryPressureListener underTest =
        MemoryPressureListener.createFromBeans(
            ImmutableList.of(mockBean, anotherMockBean), directExecutor());
    underTest.initForInvocation(
        eventBus, mock(GcThrashingDetector.class), mock(GcChurningDetector.class));
    verify(mockBean).addNotificationListener(underTest, null, null);
    verify(anotherMockBean).addNotificationListener(underTest, null, null);

    GcInfo mockGcInfo = mock(GcInfo.class);
    MemoryUsage mockMemoryUsageForTenuredSpace = mock(MemoryUsage.class);
    when(mockMemoryUsageForTenuredSpace.getUsed()).thenReturn(1L);
    when(mockMemoryUsageForTenuredSpace.getMax()).thenReturn(0L);
    MemoryUsage mockMemoryUsageForAnotherTenuredSpace = mock(MemoryUsage.class);
    when(mockMemoryUsageForAnotherTenuredSpace.getUsed()).thenReturn(2L);
    when(mockMemoryUsageForAnotherTenuredSpace.getMax()).thenReturn(3L);
    when(mockGcInfo.getMemoryUsageAfterGc())
        .thenReturn(
            ImmutableMap.of(
                TENURED_SPACE_NAME,
                mockMemoryUsageForTenuredSpace,
                anotherTenuredSpaceName,
                mockMemoryUsageForAnotherTenuredSpace));
    when(mockGcInfo.getDuration()).thenReturn(42_000L);

    Notification notification =
        new Notification(
            GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION, "test", 123);
    notification.setUserData(
        new GarbageCollectionNotificationInfo("gcName", "gcAction", "non-manual", mockGcInfo)
            .toCompositeData(null));
    underTest.handleNotification(notification, null);

    assertThat(events)
        .containsExactly(
            MemoryPressureEvent.newBuilder()
                .setWasManualGc(false)
                .setTenuredSpaceUsedBytes(2L)
                .setTenuredSpaceMaxBytes(3L)
                .setDuration(Duration.ofSeconds(42))
                .build());
  }

  @Test
  public void directlyInvokesGcThrashingDetectorAndGcChurnDetector() {
    MemoryPressureListener underTest =
        MemoryPressureListener.createFromBeans(
            ImmutableList.of(mockUselessBean, mockBean), directExecutor());
    verify(mockBean).addNotificationListener(underTest, null, null);
    verify(mockUselessBean, never()).addNotificationListener(any(), any(), any());

    GcThrashingDetector mockGcThrashingDetector = mock(GcThrashingDetector.class);
    GcChurningDetector mockGcChurningDetector = mock(GcChurningDetector.class);
    underTest.initForInvocation(eventBus, mockGcThrashingDetector, mockGcChurningDetector);

    GcInfo mockGcInfo = mock(GcInfo.class);
    String nonTenuredSpaceName = "nope";
    MemoryUsage mockMemoryUsageForNonTenuredSpace = mock(MemoryUsage.class);
    MemoryUsage mockMemoryUsageForTenuredSpace = mock(MemoryUsage.class);
    when(mockMemoryUsageForTenuredSpace.getUsed()).thenReturn(99L);
    when(mockMemoryUsageForTenuredSpace.getMax()).thenReturn(100L);
    when(mockGcInfo.getMemoryUsageAfterGc())
        .thenReturn(
            ImmutableMap.of(
                nonTenuredSpaceName,
                mockMemoryUsageForNonTenuredSpace,
                TENURED_SPACE_NAME,
                mockMemoryUsageForTenuredSpace));
    when(mockGcInfo.getDuration()).thenReturn(42_000L);

    Notification notification =
        new Notification(
            GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION, "test", 123);
    notification.setUserData(
        new GarbageCollectionNotificationInfo("gcName", "end of major GC", "non-manual", mockGcInfo)
            .toCompositeData(null));
    underTest.handleNotification(notification, null);

    MemoryPressureEvent event =
        MemoryPressureEvent.newBuilder()
            .setWasManualGc(false)
            .setWasFullGc(true)
            .setTenuredSpaceUsedBytes(99)
            .setTenuredSpaceMaxBytes(100)
            .setDuration(Duration.ofSeconds(42))
            .build();
    assertThat(events).containsExactly(event);
    verify(mockGcThrashingDetector).handle(eq(event));
    verify(mockGcChurningDetector).handle(eq(event));
  }

  @Test
  public void forwardsTargetParsingComplete() {
    MemoryPressureListener underTest =
        MemoryPressureListener.createFromBeans(
            ImmutableList.of(mockUselessBean, mockBean), directExecutor());

    GcChurningDetector mockGcChurningDetector = mock(GcChurningDetector.class);
    underTest.initForInvocation(eventBus, mock(GcThrashingDetector.class), mockGcChurningDetector);
    underTest.targetParsingComplete(42);

    verify(mockGcChurningDetector).targetParsingComplete(eq(42));
  }
}
