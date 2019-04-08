// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.profiler;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.profiler.Profiler.Format.BINARY_BAZEL_FORMAT;
import static com.google.devtools.build.lib.profiler.Profiler.Format.JSON_TRACE_FILE_FORMAT;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.profiler.Profiler.SlowTask;
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.testutil.TestUtils;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.Deflater;
import java.util.zip.DeflaterOutputStream;
import java.util.zip.Inflater;
import java.util.zip.InflaterInputStream;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for the profiler.
 */
@TestSpec(size = Suite.MEDIUM_TESTS) // testConcurrentProfiling takes ~700ms, testProfiler 100ms.
@RunWith(JUnit4.class)
public class ProfilerTest {
  private Profiler profiler = Profiler.instance();
  private ManualClock clock;

  @Before
  public final void setManualClock() {
    clock = new ManualClock();
    BlazeClock.setClock(clock);
  }

  @After
  public void forceStopToAvoidPoisoningTheProfiler() {
    // If a test does not stop the profiler, e.g., due to a test failure, all subsequent tests fail
    // because the profiler is still running, so we force-stop the profiler here.
    try {
      profiler.stop();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private ImmutableSet<ProfilerTask> getAllProfilerTasks() {
    return ImmutableSet.copyOf(ProfilerTask.values());
  }

  private ImmutableSet<ProfilerTask> getSlowestProfilerTasks() {
    ImmutableSet.Builder<ProfilerTask> profiledTasksBuilder = ImmutableSet.builder();
    for (ProfilerTask profilerTask : ProfilerTask.values()) {
      if (profilerTask.collectsSlowestInstances()) {
        profiledTasksBuilder.add(profilerTask);
      }
    }
    return profiledTasksBuilder.build();
  }

  private ByteArrayOutputStream start(ImmutableSet<ProfilerTask> tasks, Profiler.Format format)
      throws IOException {
    ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    profiler.start(
        tasks,
        buffer,
        format,
        "test",
        false,
        BlazeClock.instance(),
        BlazeClock.nanoTime(),
        /* enabledCpuUsageProfiling= */ false,
        /* slimProfile= */ false);
    return buffer;
  }

  private void startUnbuffered(ImmutableSet<ProfilerTask> tasks) throws IOException {
    profiler.start(
        tasks,
        null,
        null,
        "test",
        false,
        BlazeClock.instance(),
        BlazeClock.nanoTime(),
        /* enabledCpuUsageProfiling= */ false,
        /* slimProfile= */ false);
  }

  @Test
  public void testProfilerActivation() throws Exception {
    assertThat(profiler.isActive()).isFalse();
    start(getAllProfilerTasks(), BINARY_BAZEL_FORMAT);
    assertThat(profiler.isActive()).isTrue();

    profiler.stop();
    assertThat(profiler.isActive()).isFalse();
  }

  @Test
  public void testTaskDetails() throws Exception {
    ByteArrayOutputStream buffer = start(getAllProfilerTasks(), BINARY_BAZEL_FORMAT);
    try (SilentCloseable c = profiler.profile(ProfilerTask.ACTION, "action task")) {
      profiler.logEvent(ProfilerTask.INFO, "event");
    }
    profiler.stop();
    ProfileInfo info = ProfileInfo.loadProfile(new ByteArrayInputStream(buffer.toByteArray()));
    info.calculateStats();

    ProfileInfo.Task task = info.allTasksById.get(0);
    assertThat(task.id).isEqualTo(1);
    assertThat(task.type).isEqualTo(ProfilerTask.ACTION);
    assertThat(task.getDescription()).isEqualTo("action task");

    task = info.allTasksById.get(1);
    assertThat(task.id).isEqualTo(2);
    assertThat(task.type).isEqualTo(ProfilerTask.INFO);
    assertThat(task.getDescription()).isEqualTo("event");
  }

  @Test
  public void testProfiler() throws Exception {
    ByteArrayOutputStream buffer = start(getAllProfilerTasks(), BINARY_BAZEL_FORMAT);
    profiler.logSimpleTask(BlazeClock.instance().nanoTime(),
                           ProfilerTask.PHASE, "profiler start");
    try (SilentCloseable c = profiler.profile(ProfilerTask.ACTION, "complex task")) {
      profiler.logEvent(ProfilerTask.PHASE, "event1");
      try (SilentCloseable c2 = profiler.profile(ProfilerTask.ACTION_CHECK, "complex subtask")) {
        // next task takes less than 10 ms and should be only aggregated
        profiler.logSimpleTask(BlazeClock.instance().nanoTime(),
                               ProfilerTask.VFS_STAT, "stat1");
        long startTime = BlazeClock.instance().nanoTime();
        clock.advanceMillis(20);
        // this one will take at least 20 ms and should be present
        profiler.logSimpleTask(startTime, ProfilerTask.VFS_STAT, "stat2");
      }
    }
    profiler.stop();
    // all other calls to profiler should be ignored
    profiler.logEvent(ProfilerTask.PHASE, "should be ignored");

    ProfileInfo info = ProfileInfo.loadProfile(new ByteArrayInputStream(buffer.toByteArray()));
    info.calculateStats();
    assertThat(info.allTasksById).hasSize(6); // only 5 tasks + finalization should be recorded

    ProfileInfo.Task task = info.allTasksById.get(0);
    assertThat(task.stats.isEmpty()).isTrue();

    task = info.allTasksById.get(1);
    int count = 0;
    for (ProfileInfo.AggregateAttr attr : task.getStatAttrArray()) {
      if (attr != null) {
        count++;
      }
    }
    assertThat(count).isEqualTo(2); // only children are GENERIC and ACTION_CHECK
    assertThat(ProfilerTask.TASK_COUNT).isEqualTo(task.aggregatedStats.toArray().length);
    assertThat(task.aggregatedStats.getAttr(ProfilerTask.VFS_STAT).count).isEqualTo(2);

    task = info.allTasksById.get(2);
    assertThat(task.durationNanos).isEqualTo(0);

    task = info.allTasksById.get(3);
    assertThat(task.stats.getAttr(ProfilerTask.VFS_STAT).count).isEqualTo(2);
    assertThat(task.subtasks).hasLength(1);
    assertThat(task.subtasks[0].getDescription()).isEqualTo("stat2");
    // assert that startTime grows with id
    long time = -1;
    for (ProfileInfo.Task t : info.allTasksById) {
      assertThat(t.startTime).isAtLeast(time);
      time = t.startTime;
    }
  }

  @Test
  public void testProfilerRecordingAllEvents() throws Exception {
    ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    profiler.start(
        getAllProfilerTasks(),
        buffer,
        BINARY_BAZEL_FORMAT,
        "basic test",
        true,
        BlazeClock.instance(),
        BlazeClock.instance().nanoTime(),
        /* enabledCpuUsageProfiling= */ false,
        /* slimProfile= */ false);
    try (SilentCloseable c = profiler.profile(ProfilerTask.ACTION, "action task")) {
      // Next task takes less than 10 ms but should be recorded anyway.
      clock.advanceMillis(1);
      profiler.logSimpleTask(BlazeClock.instance().nanoTime(), ProfilerTask.VFS_STAT, "stat1");
    }
    profiler.stop();
    ProfileInfo info = ProfileInfo.loadProfile(new ByteArrayInputStream(buffer.toByteArray()));
    info.calculateStats();
    assertThat(info.allTasksById).hasSize(3); // 2 tasks + finalization should be recorded

    ProfileInfo.Task task = info.allTasksById.get(1);
    assertThat(task.type).isEqualTo(ProfilerTask.VFS_STAT);

    // Check that task would have been dropped if profiler was not configured to record everything.
    assertThat(task.durationNanos).isLessThan(ProfilerTask.VFS_STAT.minDuration);
  }

  @Test
  public void testProfilerRecordingOnlySlowestEvents() throws Exception {
    ByteArrayOutputStream buffer = new ByteArrayOutputStream();

    profiler.start(
        getSlowestProfilerTasks(),
        buffer,
        BINARY_BAZEL_FORMAT,
        "test",
        true,
        BlazeClock.instance(),
        BlazeClock.instance().nanoTime(),
        /* enabledCpuUsageProfiling= */ false,
        /* slimProfile= */ false);
    profiler.logSimpleTask(10000, 20000, ProfilerTask.VFS_STAT, "stat");
    profiler.logSimpleTask(20000, 30000, ProfilerTask.REMOTE_EXECUTION, "remote execution");

    assertThat(profiler.isProfiling(ProfilerTask.VFS_STAT)).isTrue();
    assertThat(profiler.isProfiling(ProfilerTask.REMOTE_EXECUTION)).isFalse();

    profiler.stop();

    ProfileInfo info = ProfileInfo.loadProfile(new ByteArrayInputStream(buffer.toByteArray()));
    info.calculateStats();
    assertThat(info.allTasksById).hasSize(1); // only VFS_STAT task should be recorded

    ProfileInfo.Task task = info.allTasksById.get(0);
    assertThat(task.type).isEqualTo(ProfilerTask.VFS_STAT);
  }

  @Test
  public void testSlowestTasks() throws Exception {
    startUnbuffered(getAllProfilerTasks());
    profiler.logSimpleTaskDuration(
        Profiler.nanoTimeMaybe(), Duration.ofSeconds(10), ProfilerTask.LOCAL_PARSE, "foo");
    Iterable<SlowTask> slowestTasks = profiler.getSlowestTasks();
    assertThat(slowestTasks).hasSize(1);
    SlowTask task = slowestTasks.iterator().next();
    assertThat(task.type).isEqualTo(ProfilerTask.LOCAL_PARSE);
    profiler.stop();
  }

  @Test
  public void testGetSlowestTasksCapped() throws Exception {
    startUnbuffered(getSlowestProfilerTasks());

    // Add some fast tasks - these shouldn't show up in the slowest.
    for (int i = 0; i < ProfilerTask.VFS_STAT.slowestInstancesCount; i++) {
      profiler.logSimpleTask(
          /*startTimeNanos=*/ 1,
          /*stopTimeNanos=*/ ProfilerTask.VFS_STAT.minDuration + 10,
          ProfilerTask.VFS_STAT,
          "stat");
    }

    // Add some slow tasks we expect to show up in the slowest.
    List<Long> expectedSlowestDurations = new ArrayList<>();
    for (int i = 0; i < ProfilerTask.VFS_STAT.slowestInstancesCount; i++) {
      long fakeDuration = ProfilerTask.VFS_STAT.minDuration + i + 10_000;
      profiler.logSimpleTask(
          /*startTimeNanos=*/ 1,
          /*stopTimeNanos=*/ fakeDuration + 1,
          ProfilerTask.VFS_STAT,
          "stat");
      expectedSlowestDurations.add(fakeDuration);
    }

    // Sprinkle in a whole bunch of fast tasks from different thread ids - necessary because
    // internally aggregation is sharded across several aggregators, sharded by thread id.
    // It's possible all these threads wind up in the same shard, we'll take our chances.
    ImmutableList.Builder<Thread> threadsBuilder = ImmutableList.builder();
    try {
      for (int i = 0; i < 32; i++) {
        Thread thread = new Thread() {
          @Override
          public void run() {
            for (int j = 0; j < 100; j++) {
              profiler.logSimpleTask(
                  /*startTimeNanos=*/ 1,
                  /*stopTimeNanos=*/ ProfilerTask.VFS_STAT.minDuration + j + 1,
                  ProfilerTask.VFS_STAT,
                  "stat");
            }
          }
        };
        threadsBuilder.add(thread);
        thread.start();
      }
    } finally {
      threadsBuilder.build().forEach(
          t -> {
            try {
              t.join(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
            } catch (InterruptedException e) {
              t.interrupt();
              // This'll go ahead and interrupt all the others. The thread we just interrupted is
              // lightweight enough that it's reasonable to assume it'll exit.
              Thread.currentThread().interrupt();
            }
          });
    }

    ImmutableList<SlowTask> slowTasks = ImmutableList.copyOf(profiler.getSlowestTasks());
    assertThat(slowTasks).hasSize(ProfilerTask.VFS_STAT.slowestInstancesCount);

    ImmutableList<Long> slowestDurations = slowTasks.stream()
        .map(task -> task.getDurationNanos())
        .collect(ImmutableList.toImmutableList());
    assertThat(slowestDurations).containsExactlyElementsIn(expectedSlowestDurations);
  }

  @Test
  public void testProfilerRecordsNothing() throws Exception {
    ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    profiler.start(
        ImmutableSet.of(),
        buffer,
        BINARY_BAZEL_FORMAT,
        "test",
        true,
        BlazeClock.instance(),
        BlazeClock.instance().nanoTime(),
        /* enabledCpuUsageProfiling= */ false,
        /* slimProfile= */ false);
    profiler.logSimpleTask(10000, 20000, ProfilerTask.VFS_STAT, "stat");

    assertThat(ProfilerTask.VFS_STAT.collectsSlowestInstances()).isTrue();
    assertThat(profiler.isProfiling(ProfilerTask.VFS_STAT)).isFalse();

    profiler.stop();

    ProfileInfo info = ProfileInfo.loadProfile(new ByteArrayInputStream(buffer.toByteArray()));
    info.calculateStats();
    assertThat(info.allTasksById).isEmpty();
  }

  @Test
  public void testConcurrentProfiling() throws Exception {
    ByteArrayOutputStream buffer = start(getAllProfilerTasks(), BINARY_BAZEL_FORMAT);

    long id = Thread.currentThread().getId();
    Thread thread1 = new Thread() {
      @Override public void run() {
        for (int i = 0; i < 10000; i++) {
          Profiler.instance().logEvent(ProfilerTask.INFO, "thread1");
        }
      }
    };
    long id1 = thread1.getId();
    Thread thread2 = new Thread() {
      @Override public void run() {
        for (int i = 0; i < 10000; i++) {
          Profiler.instance().logEvent(ProfilerTask.INFO, "thread2");
        }
      }
    };
    long id2 = thread2.getId();

    try (SilentCloseable c = profiler.profile(ProfilerTask.PHASE, "main task")) {
      profiler.logEvent(ProfilerTask.INFO, "starting threads");
      thread1.start();
      thread2.start();
      thread2.join();
      thread1.join();
      profiler.logEvent(ProfilerTask.INFO, "joined");
    }
    profiler.stop();

    ProfileInfo info = ProfileInfo.loadProfile(new ByteArrayInputStream(buffer.toByteArray()));
    info.calculateStats();
    info.analyzeRelationships();
    assertThat(info.allTasksById).hasSize(4 + 10000 + 10000); // total number of tasks
    assertThat(info.tasksByThread).hasSize(3); // total number of threads
    // while main thread had 3 tasks, 2 of them were nested, so tasksByThread
    // would contain only one "main task" task
    assertThat(info.tasksByThread.get(id)).hasLength(2);
    ProfileInfo.Task mainTask = info.tasksByThread.get(id)[0];
    assertThat(mainTask.getDescription()).isEqualTo("main task");
    assertThat(mainTask.subtasks).hasLength(2);
    // other threads had 10000 independent recorded tasks each
    assertThat(info.tasksByThread.get(id1)).hasLength(10000);
    assertThat(info.tasksByThread.get(id2)).hasLength(10000);
    int startId = mainTask.subtasks[0].id; // id of "starting threads"
    int endId = mainTask.subtasks[1].id; // id of "joining"
    assertThat(startId).isLessThan(info.tasksByThread.get(id1)[0].id);
    assertThat(startId).isLessThan(info.tasksByThread.get(id2)[0].id);
    assertThat(endId).isGreaterThan(info.tasksByThread.get(id1)[9999].id);
    assertThat(endId).isGreaterThan(info.tasksByThread.get(id2)[9999].id);
  }

  @Test
  public void testPhaseTasks() throws Exception {
    ByteArrayOutputStream buffer = start(getAllProfilerTasks(), BINARY_BAZEL_FORMAT);
    Thread thread1 = new Thread() {
      @Override public void run() {
        for (int i = 0; i < 100; i++) {
          Profiler.instance().logEvent(ProfilerTask.INFO, "thread1");
        }
      }
    };
    profiler.markPhase(ProfilePhase.INIT); // Empty phase.
    profiler.markPhase(ProfilePhase.LOAD);
    thread1.start();
    thread1.join();
    clock.advanceMillis(1);
    profiler.markPhase(ProfilePhase.ANALYZE);
    Thread thread2 =
        new Thread() {
          @Override
          public void run() {
            try (SilentCloseable c = profiler.profile(ProfilerTask.INFO, "complex task")) {
              for (int i = 0; i < 100; i++) {
                Profiler.instance().logEvent(ProfilerTask.INFO, "thread2a");
              }
            }
            try {
              profiler.markPhase(ProfilePhase.EXECUTE);
            } catch (InterruptedException e) {
              throw new IllegalStateException(e);
            }
            for (int i = 0; i < 100; i++) {
              Profiler.instance().logEvent(ProfilerTask.INFO, "thread2b");
            }
          }
        };
    thread2.start();
    thread2.join();
    profiler.logEvent(ProfilerTask.INFO, "last task");
    clock.advanceMillis(1);
    profiler.stop();

    ProfileInfo info = ProfileInfo.loadProfile(new ByteArrayInputStream(buffer.toByteArray()));
    info.calculateStats();
    info.analyzeRelationships();
    // number of tasks: INIT(1) + LOAD(1) + Thread1.TEST(100) + ANALYZE(1)
    // + Thread2a.TEST(100) + TEST(1) + EXECUTE(1) + Thread2b.TEST(100) + TEST(1) + INFO(1)
    assertThat(info.allTasksById).hasSize(1 + 1 + 100 + 1 + 100 + 1 + 1 + 100 + 1 + 1);
    assertThat(info.tasksByThread).hasSize(3); // total number of threads
    // Phase0 contains only itself
    ProfileInfo.Task p0 = info.getPhaseTask(ProfilePhase.INIT);
    assertThat(info.getTasksForPhase(p0)).hasSize(1);
    // Phase1 contains itself and 100 TEST "thread1" tasks
    ProfileInfo.Task p1 = info.getPhaseTask(ProfilePhase.LOAD);
    assertThat(info.getTasksForPhase(p1)).hasSize(101);
    // Phase2 contains itself and 1 "complex task"
    ProfileInfo.Task p2 = info.getPhaseTask(ProfilePhase.ANALYZE);
    assertThat(info.getTasksForPhase(p2)).hasSize(2);
    // Phase3 contains itself, 100 TEST "thread2b" tasks and "last task"
    ProfileInfo.Task p3 = info.getPhaseTask(ProfilePhase.EXECUTE);
    assertThat(info.getTasksForPhase(p3)).hasSize(103);
  }

  @Test
  public void testCorruptedFile() throws Exception {
    ByteArrayOutputStream buffer = start(getAllProfilerTasks(), BINARY_BAZEL_FORMAT);
    for (int i = 0; i < 100; i++) {
      try (SilentCloseable c = profiler.profile(ProfilerTask.INFO, "outer task " + i)) {
        clock.advanceMillis(1);
        profiler.logEvent(ProfilerTask.INFO, "inner task " + i);
      }
    }
    profiler.stop();

    ProfileInfo info = ProfileInfo.loadProfile(new ByteArrayInputStream(buffer.toByteArray()));
    info.calculateStats();
    assertThat(info.isCorruptedOrIncomplete()).isFalse();

    info = ProfileInfo.loadProfile(
        new ByteArrayInputStream(Arrays.copyOf(buffer.toByteArray(), 2000)));
    info.calculateStats();
    assertThat(info.isCorruptedOrIncomplete()).isTrue();
    // Since root tasks will appear after nested tasks in the profile file and
    // we have exactly one nested task for each root task, the following will always
    // be true for our corrupted file:
    // 0 <= number_of_all_tasks - 2*number_of_root_tasks <= 1
    assertThat(info.allTasksById.size() / 2).isEqualTo(info.rootTasksById.size());
  }

  @Test
  public void testUnsupportedProfilerRecord() throws Exception {
    ByteArrayOutputStream buffer = start(getAllProfilerTasks(), BINARY_BAZEL_FORMAT);
    try (SilentCloseable c = profiler.profile(ProfilerTask.INFO, "outer task")) {
      profiler.logEvent(ProfilerTask.PHASE, "inner task");
    }
    try (SilentCloseable c = profiler.profile(ProfilerTask.SCANNER, "outer task 2")) {
      profiler.logSimpleTask(Profiler.nanoTimeMaybe(), ProfilerTask.INFO, "inner task 2");
    }
    profiler.stop();

    // Validate our test profile.
    ProfileInfo info = ProfileInfo.loadProfile(new ByteArrayInputStream(buffer.toByteArray()));
    info.calculateStats();
    assertThat(info.isCorruptedOrIncomplete()).isFalse();
    assertThat(info.getStatsForType(ProfilerTask.INFO, info.rootTasksById).count).isEqualTo(3);
    assertThat(info.getStatsForType(ProfilerTask.UNKNOWN, info.rootTasksById).count).isEqualTo(0);

    // Now replace "TEST" type with something unsupported - e.g. "XXXX".
    byte[] deflated = ByteStreams.toByteArray(new InflaterInputStream(
        new ByteArrayInputStream(buffer.toByteArray()), new Inflater(false), 65536));
    String content = new String(deflated, ISO_8859_1);
    int infoIndex = content.indexOf("INFO");
    assertThat(infoIndex).isGreaterThan(0);
    content = content.substring(0, infoIndex) + "XXXX" + content.substring(infoIndex + 4);

    buffer = new ByteArrayOutputStream();
    OutputStream out =
        new DeflaterOutputStream(buffer, new Deflater(Deflater.BEST_SPEED, false), 65536);
    out.write(content.getBytes(ISO_8859_1));
    out.close();

    // Validate that XXXX records were classified as UNKNOWN.
    info = ProfileInfo.loadProfile(new ByteArrayInputStream(buffer.toByteArray()));
    info.calculateStats();
    assertThat(info.isCorruptedOrIncomplete()).isFalse();
    assertThat(info.getStatsForType(ProfilerTask.INFO, info.rootTasksById).count).isEqualTo(0);
    assertThat(info.getStatsForType(ProfilerTask.SCANNER, info.rootTasksById).count).isEqualTo(1);
    assertThat(info.getStatsForType(ProfilerTask.PHASE, info.rootTasksById).count).isEqualTo(1);
    assertThat(info.getStatsForType(ProfilerTask.UNKNOWN, info.rootTasksById).count).isEqualTo(3);
  }

  @Test
  public void testResilenceToNonDecreasingNanoTimes() throws Exception {
    final long initialNanoTime = BlazeClock.instance().nanoTime();
    final AtomicInteger numNanoTimeCalls = new AtomicInteger(0);
    Clock badClock = new Clock() {
      @Override
      public long currentTimeMillis() {
        return BlazeClock.instance().currentTimeMillis();
      }

      @Override
      public long nanoTime() {
        return initialNanoTime - numNanoTimeCalls.addAndGet(1);
      }
    };
    profiler.start(
        getAllProfilerTasks(),
        new ByteArrayOutputStream(),
        BINARY_BAZEL_FORMAT,
        "testResilenceToNonDecreasingNanoTimes",
        false,
        badClock,
        initialNanoTime,
        /* enabledCpuUsageProfiling= */ false,
        /* slimProfile= */ false);
    profiler.logSimpleTask(badClock.nanoTime(), ProfilerTask.INFO, "some task");
    profiler.stop();
  }

  /** Checks that the histograms are cleared in the stop call. */
  @Test
  public void testEmptyTaskHistograms() throws Exception {
    startUnbuffered(getAllProfilerTasks());
    profiler.logSimpleTaskDuration(
        Profiler.nanoTimeMaybe(), Duration.ofSeconds(10), ProfilerTask.INFO, "foo");
    profiler.stop();
    ImmutableList<StatRecorder> histograms = profiler.getTasksHistograms();
    for (StatRecorder recorder : histograms) {
      assertThat(recorder.isEmpty()).isTrue();
    }
  }

  @Test
  public void testTaskHistograms() throws Exception {
    startUnbuffered(getAllProfilerTasks());
    profiler.logSimpleTaskDuration(
        Profiler.nanoTimeMaybe(), Duration.ofSeconds(10), ProfilerTask.INFO, "foo");
    ImmutableList<StatRecorder> histograms = profiler.getTasksHistograms();
    StatRecorder infoStatRecorder = histograms.get(ProfilerTask.INFO.ordinal());
    assertThat(infoStatRecorder.isEmpty()).isFalse();
    // This is the only provided API to get the contents of the StatRecorder.
    assertThat(infoStatRecorder.toString()).contains("'INFO'");
    assertThat(infoStatRecorder.toString()).contains("Count: 1");
    assertThat(infoStatRecorder.toString()).contains("[8192..16384 ms]");
    // The stop() call is here because the histograms are cleared in the stop call. See the
    // documentation of {@link Profiler#getTasksHistograms}.
    profiler.stop();
  }

  @Test
  public void testIOExceptionInOutputStreamBinaryFormat() throws Exception {
    OutputStream failingOutputStream = new OutputStream() {
      @Override
      public void write(int b) throws IOException {
        throw new IOException("Expected failure.");
      }
    };
    profiler.start(
        getAllProfilerTasks(),
        failingOutputStream,
        BINARY_BAZEL_FORMAT,
        "basic test",
        false,
        BlazeClock.instance(),
        BlazeClock.instance().nanoTime(),
        /* enabledCpuUsageProfiling= */ false,
        /* slimProfile= */ false);
    profiler.logSimpleTaskDuration(
        Profiler.nanoTimeMaybe(), Duration.ofSeconds(10), ProfilerTask.INFO, "foo");
    try {
      profiler.stop();
      fail();
    } catch (IOException expected) {
      assertThat(expected).hasMessageThat().isEqualTo("Expected failure.");
    }
  }

  @Test
  public void testIOExceptionInOutputStreamJsonFormat() throws Exception {
    OutputStream failingOutputStream = new OutputStream() {
      @Override
      public void write(int b) throws IOException {
        throw new IOException("Expected failure.");
      }
    };
    profiler.start(
        getAllProfilerTasks(),
        failingOutputStream,
        JSON_TRACE_FILE_FORMAT,
        "basic test",
        false,
        BlazeClock.instance(),
        BlazeClock.instance().nanoTime(),
        /* enabledCpuUsageProfiling= */ false,
        /* slimProfile= */ false);
    profiler.logSimpleTaskDuration(
        Profiler.nanoTimeMaybe(), Duration.ofSeconds(10), ProfilerTask.INFO, "foo");
    try {
      profiler.stop();
      fail();
    } catch (IOException expected) {
      assertThat(expected).hasMessageThat().isEqualTo("Expected failure.");
    }
  }

  private ByteArrayOutputStream getJsonProfileOutputStream(boolean slimProfile) throws IOException {
    ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
    profiler.start(
        getAllProfilerTasks(),
        outputStream,
        JSON_TRACE_FILE_FORMAT,
        "basic test",
        false,
        BlazeClock.instance(),
        BlazeClock.instance().nanoTime(),
        /* enabledCpuUsageProfiling= */ false,
        slimProfile);
    long curTime = Profiler.nanoTimeMaybe();
    for (int i = 0; i < 100_000; i++) {
      Duration duration;
      if (i % 100 == 0) {
        duration = Duration.ofSeconds(1);
      } else {
        duration = Duration.ofMillis(i % 250);
      }
      profiler.logSimpleTaskDuration(curTime, duration, ProfilerTask.INFO, "foo");
      curTime += duration.toNanos();
    }
    profiler.stop();
    return outputStream;
  }

  @Test
  public void testSlimProfileSize() throws Exception {
    ByteArrayOutputStream fatOutputStream = getJsonProfileOutputStream(/* slimProfile= */ false);
    String fatOutput = fatOutputStream.toString();
    assertThat(fatOutput).doesNotContain("merged");

    ByteArrayOutputStream slimOutputStream = getJsonProfileOutputStream(/* slimProfile= */ true);
    String slimOutput = slimOutputStream.toString();
    assertThat(slimOutput).contains("merged");

    long fatProfileLen = fatOutputStream.size();
    long slimProfileLen = slimOutputStream.size();
    assertThat(fatProfileLen).isAtLeast(8 * slimProfileLen);

    long fatProfileLineCount = fatOutput.split("\n").length;
    long slimProfileLineCount = slimOutput.split("\n").length;
    assertThat(fatProfileLineCount).isAtLeast(8 * slimProfileLineCount);
  }
}
