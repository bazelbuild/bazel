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

package com.google.devtools.build.lib.metrics;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.withSettings;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.metrics.PostGCMemoryUseRecorder.PeakHeap;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.sun.management.GarbageCollectionNotificationInfo;
import com.sun.management.GcInfo;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.MemoryUsage;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import javax.management.Notification;
import javax.management.NotificationEmitter;
import javax.management.NotificationFilter;
import javax.management.NotificationListener;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link PostGCMemoryUseRecorder}. */
@RunWith(JUnit4.class)
public final class PostGCMemoryUseRecorderTest {

  private final ManualClock clock = new ManualClock();

  @Test
  public void listenToSingleNonCopyGC() {
    List<GarbageCollectorMXBean> beans = createGCBeans(new String[] {"FooGC"});

    PostGCMemoryUseRecorder rec = new PostGCMemoryUseRecorder(beans, BugReporter.defaultInstance());
    verify((NotificationEmitter) beans.get(0), times(1)).addNotificationListener(rec, null, null);
  }

  @Test
  public void listenToMultipleNonCopyGCs() {
    List<GarbageCollectorMXBean> beans = createGCBeans(new String[] {"FooGC", "BarGC"});

    PostGCMemoryUseRecorder rec = new PostGCMemoryUseRecorder(beans, BugReporter.defaultInstance());
    verify((NotificationEmitter) beans.get(0), times(1)).addNotificationListener(rec, null, null);
    verify((NotificationEmitter) beans.get(1), times(1)).addNotificationListener(rec, null, null);
  }

  @Test
  public void dontListenToCopyGC() {
    List<GarbageCollectorMXBean> beans = createGCBeans(new String[] {"FooGC", "Copy"});

    PostGCMemoryUseRecorder rec = new PostGCMemoryUseRecorder(beans, BugReporter.defaultInstance());
    verify((NotificationEmitter) beans.get(0), times(1)).addNotificationListener(rec, null, null);
    verify((NotificationEmitter) beans.get(1), never())
        .addNotificationListener(
            any(NotificationListener.class), any(NotificationFilter.class), any());
  }

  @Test
  public void peakHeapsStartAbsent() {
    PostGCMemoryUseRecorder rec =
        new PostGCMemoryUseRecorder(new ArrayList<>(), BugReporter.defaultInstance());
    assertThat(rec.getPeakPostGcHeap()).isEmpty();
    assertThat(rec.getPeakPostGcHeapTenuredSpace()).isEmpty();
  }

  @Test
  public void peakHeapsAbsentAfterReset() {
    PostGCMemoryUseRecorder rec =
        new PostGCMemoryUseRecorder(new ArrayList<>(), BugReporter.defaultInstance());
    rec.handleNotification(
        createOneTenuredSpaceOneNonTenuredSpaceMajorGCNotifications(1000L), null);
    rec.reset();
    assertThat(rec.getPeakPostGcHeap()).isEmpty();
    assertThat(rec.getPeakPostGcHeapTenuredSpace()).isEmpty();
  }

  @Test
  public void noGcCauseEventsNotIgnored() {
    PostGCMemoryUseRecorder underTest =
        new PostGCMemoryUseRecorder(ImmutableList.of(), BugReporter.defaultInstance());
    Notification notificationWithNoGcCause =
        createMockNotification(
            GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION,
            "end of major GC",
            /*cause=*/ "No GC",
            ImmutableMap.of("somepool", 100L),
            /*memUsedBefore=*/ null);

    underTest.handleNotification(notificationWithNoGcCause, /*handback=*/ null);

    assertThat(underTest.getPeakPostGcHeap())
        .hasValue(PeakHeap.create(100, clock.currentTimeMillis()));
    assertThat(underTest.getPeakPostGcHeapTenuredSpace())
        .hasValue(PeakHeap.create(0, clock.currentTimeMillis()));
  }

  @Test
  public void peakHeapsIncreaseWhenBigger() {
    PostGCMemoryUseRecorder rec =
        new PostGCMemoryUseRecorder(new ArrayList<>(), BugReporter.defaultInstance());

    clock.advanceMillis(1);
    rec.handleNotification(
        createOneTenuredSpaceOneNonTenuredSpaceMajorGCNotifications(1000L), null);
    assertThat(rec.getPeakPostGcHeap()).hasValue(PeakHeap.create(2000, clock.currentTimeMillis()));
    assertThat(rec.getPeakPostGcHeapTenuredSpace())
        .hasValue(PeakHeap.create(1000, clock.currentTimeMillis()));

    clock.advanceMillis(1);
    rec.handleNotification(
        createOneTenuredSpaceOneNonTenuredSpaceMajorGCNotifications(1001L), null);
    assertThat(rec.getPeakPostGcHeap()).hasValue(PeakHeap.create(2002, clock.currentTimeMillis()));
    assertThat(rec.getPeakPostGcHeapTenuredSpace())
        .hasValue(PeakHeap.create(1001, clock.currentTimeMillis()));

    clock.advanceMillis(1);
    rec.handleNotification(
        createOneTenuredSpaceOneNonTenuredSpaceMajorGCNotifications(1002L), null);
    assertThat(rec.getPeakPostGcHeap()).hasValue(PeakHeap.create(2004, clock.currentTimeMillis()));
    assertThat(rec.getPeakPostGcHeapTenuredSpace())
        .hasValue(PeakHeap.create(1002, clock.currentTimeMillis()));
  }

  @Test
  public void peakHeapsDontDecrease() {
    PostGCMemoryUseRecorder rec =
        new PostGCMemoryUseRecorder(new ArrayList<>(), BugReporter.defaultInstance());

    clock.advanceMillis(1);
    rec.handleNotification(createOneTenuredSpaceOneNonTenuredSpaceMajorGCNotifications(1000), null);
    PeakHeap expectedTotal = PeakHeap.create(2000, clock.currentTimeMillis());
    PeakHeap expectedTenuredSpace = PeakHeap.create(1000, clock.currentTimeMillis());

    clock.advanceMillis(1);
    rec.handleNotification(createOneTenuredSpaceOneNonTenuredSpaceMajorGCNotifications(500), null);
    assertThat(rec.getPeakPostGcHeap()).hasValue(expectedTotal);
    assertThat(rec.getPeakPostGcHeapTenuredSpace()).hasValue(expectedTenuredSpace);

    clock.advanceMillis(1);
    rec.handleNotification(createOneTenuredSpaceOneNonTenuredSpaceMajorGCNotifications(999), null);
    assertThat(rec.getPeakPostGcHeap()).hasValue(expectedTotal);
    assertThat(rec.getPeakPostGcHeapTenuredSpace()).hasValue(expectedTenuredSpace);
  }

  @Test
  public void ignoreNonGCNotification() {
    PostGCMemoryUseRecorder rec =
        new PostGCMemoryUseRecorder(new ArrayList<>(), BugReporter.defaultInstance());
    rec.handleNotification(
        createMockNotification(
            "some other notification", "end of major GC", ImmutableMap.of("Foo", 1000L)),
        null);
    assertThat(rec.getPeakPostGcHeap()).isEmpty();
    assertThat(rec.getPeakPostGcHeapTenuredSpace()).isEmpty();
  }

  @Test
  public void ignoreNonMajorGCNotification() {
    PostGCMemoryUseRecorder rec =
        new PostGCMemoryUseRecorder(new ArrayList<>(), BugReporter.defaultInstance());
    rec.handleNotification(
        createMockNotification(
            GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION,
            "end of minor GC",
            ImmutableMap.of("Foo", 1000L)),
        null);
    assertThat(rec.getPeakPostGcHeap()).isEmpty();
    assertThat(rec.getPeakPostGcHeapTenuredSpace()).isEmpty();
  }

  @Test
  public void sumMemUsageInfo() {
    PostGCMemoryUseRecorder rec =
        new PostGCMemoryUseRecorder(new ArrayList<>(), BugReporter.defaultInstance());
    rec.handleNotification(
        createMajorGCNotification(
            ImmutableMap.of("Foo", 111L, "Bar", 222L, "Qux", 333L, "CMS Old Gen", 111L)),
        null);
    assertThat(rec.getPeakPostGcHeap()).hasValue(PeakHeap.create(777, clock.currentTimeMillis()));
    assertThat(rec.getPeakPostGcHeapTenuredSpace())
        .hasValue(PeakHeap.create(111, clock.currentTimeMillis()));
  }

  @Test
  public void memoryUsageReportedZeroGetsSetAndStaysSet() {
    PostGCMemoryUseRecorder rec =
        new PostGCMemoryUseRecorder(new ArrayList<>(), BugReporter.defaultInstance());
    assertThat(rec.wasMemoryUsageReportedZero()).isFalse();
    rec.handleNotification(
        createMajorGCNotification(ImmutableMap.of("Foo", 0L, "Bar", 0L, "Qux", 0L)), null);
    assertThat(rec.wasMemoryUsageReportedZero()).isTrue();
    rec.handleNotification(
        createMajorGCNotification(ImmutableMap.of("Foo", 123L, "Bar", 456L, "Qux", 789L)), null);
    assertThat(rec.wasMemoryUsageReportedZero()).isTrue();
  }

  @Test
  public void memoryUsageReportedZeroDoesntGetSet() {
    PostGCMemoryUseRecorder rec =
        new PostGCMemoryUseRecorder(new ArrayList<>(), BugReporter.defaultInstance());
    assertThat(rec.wasMemoryUsageReportedZero()).isFalse();
    rec.handleNotification(
        createMajorGCNotification(ImmutableMap.of("Foo", 123L, "Bar", 456L, "Qux", 789L)), null);
    assertThat(rec.wasMemoryUsageReportedZero()).isFalse();
  }

  @Test
  public void totalGarbageReported() {
    PostGCMemoryUseRecorder rec =
        new PostGCMemoryUseRecorder(new ArrayList<>(), BugReporter.defaultInstance());
    assertThat(rec.getGarbageStats()).isEmpty();

    rec.handleNotification(
        createMockNotification(
            GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION,
            "action",
            "cause",
            ImmutableMap.of("old", 1000L, "young", 2000L),
            ImmutableMap.of("old", 5000L, "young", 10000L)),
        /* handback= */ null);
    assertThat(rec.getGarbageStats()).containsExactly("old", 4000L, "young", 8000L);
    rec.handleNotification(
        createMockNotification(
            GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION,
            "action",
            "cause",
            ImmutableMap.of("young", 15000L),
            ImmutableMap.of("young", 20000L)),
        /* handback= */ null);
    assertThat(rec.getGarbageStats()).containsExactly("old", 4000L, "young", 13000L);

    rec.reset();
    assertThat(rec.getGarbageStats()).isEmpty();
  }

  @Test
  public void moreThanOneTenuredSpaceEventReportsBug() {
    BugReporter bugReporter = mock(BugReporter.class);
    PostGCMemoryUseRecorder rec = new PostGCMemoryUseRecorder(new ArrayList<>(), bugReporter);
    rec.handleNotification(
        createMajorGCNotification(ImmutableMap.of("CMS Old Gen", 111L, "PS Old Gen", 111L)), null);
    verify(bugReporter)
        .sendBugReport(
            argThat(
                e ->
                    e.getMessage()
                        .contains(
                            "More than one tenured space event was recorded during garbage"
                                + " collection.")));
  }

  private static GarbageCollectorMXBean createMXBeanWithName(String name) {
    GarbageCollectorMXBean b =
        mock(
            GarbageCollectorMXBean.class,
            withSettings().extraInterfaces(NotificationEmitter.class));
    when(b.getName()).thenReturn(name);
    return b;
  }

  private static List<GarbageCollectorMXBean> createGCBeans(String[] names) {
    List<GarbageCollectorMXBean> beans = new ArrayList<>();
    for (String n : names) {
      beans.add(createMXBeanWithName(n));
    }
    return beans;
  }

  private static MemoryUsage createMockMemoryUsage(long used) {
    MemoryUsage mu = mock(MemoryUsage.class);
    when(mu.getUsed()).thenReturn(used);
    return mu;
  }

  private static ImmutableMap<String, MemoryUsage> createMemoryUsageMap(Map<String, Long> memUsed) {
    ImmutableMap.Builder<String, MemoryUsage> memUsageMap = ImmutableMap.builder();
    for (Map.Entry<String, Long> e : memUsed.entrySet()) {
      memUsageMap.put(e.getKey(), createMockMemoryUsage(e.getValue()));
    }
    return memUsageMap.build();
  }

  private Notification createMockNotification(
      String type, String action, Map<String, Long> memUsed) {
    return createMockNotification(type, action, "dummycause", memUsed, /* memUsedBefore= */ null);
  }

  private Notification createMockNotification(
      String type,
      String action,
      String cause,
      Map<String, Long> memUsed,
      @Nullable Map<String, Long> memUsedBefore) {
    GcInfo gcInfo = mock(GcInfo.class);
    ImmutableMap<String, MemoryUsage> memoryUsageMap = createMemoryUsageMap(memUsed);
    when(gcInfo.getMemoryUsageAfterGc()).thenReturn(memoryUsageMap);
    if (memUsedBefore != null) {
      ImmutableMap<String, MemoryUsage> memoryUsageBeforeMap = createMemoryUsageMap(memUsedBefore);
      when(gcInfo.getMemoryUsageBeforeGc()).thenReturn(memoryUsageBeforeMap);
    }

    GarbageCollectionNotificationInfo notInfo =
        new GarbageCollectionNotificationInfo("DummyGCName", action, cause, gcInfo);

    Notification n = mock(Notification.class);
    when(n.getType()).thenReturn(type);
    when(n.getUserData()).thenReturn(notInfo.toCompositeData(null));
    when(n.getTimeStamp()).thenReturn(clock.currentTimeMillis());
    return n;
  }

  private Notification createMajorGCNotification(Map<String, Long> memUsed) {
    return createMockNotification(
        GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION,
        "end of major GC",
        memUsed);
  }

  private Notification createOneTenuredSpaceOneNonTenuredSpaceMajorGCNotifications(long used) {
    return createMajorGCNotification(ImmutableMap.of("Foo", used, "CMS Old Gen", used));
  }
}
