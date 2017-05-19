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
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.profiler.Profiler.ProfiledTaskKinds;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.Deflater;
import java.util.zip.DeflaterOutputStream;
import java.util.zip.Inflater;
import java.util.zip.InflaterInputStream;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for the profiler.
 */
@TestSpec(size = Suite.MEDIUM_TESTS) // testConcurrentProfiling takes ~700ms, testProfiler 100ms.
@RunWith(JUnit4.class)
public class ProfilerTest extends FoundationTestCase {

  private Path cacheDir;
  private Profiler profiler = Profiler.instance();

  @Before
  public final void createCacheDirectory() throws Exception {
    cacheDir = scratch.dir("/tmp");
  }

  @Test
  public void testProfilerActivation() throws Exception {
    Path cacheFile = cacheDir.getRelative("profile1.dat");
    assertFalse(profiler.isActive());
    profiler.start(ProfiledTaskKinds.ALL, cacheFile.getOutputStream(), "basic test", false,
        BlazeClock.instance(), BlazeClock.instance().nanoTime());
    assertTrue(profiler.isActive());

    profiler.stop();
    assertFalse(profiler.isActive());
  }

  @Test
  public void testTaskDetails() throws Exception {
    Path cacheFile = cacheDir.getRelative("profile1.dat");
    profiler.start(ProfiledTaskKinds.ALL, cacheFile.getOutputStream(), "basic test", false,
        BlazeClock.instance(), BlazeClock.instance().nanoTime());
    profiler.startTask(ProfilerTask.ACTION, "action task");
    profiler.logEvent(ProfilerTask.TEST, "event");
    profiler.completeTask(ProfilerTask.ACTION);
    profiler.stop();
    ProfileInfo info = ProfileInfo.loadProfile(cacheFile);
    info.calculateStats();

    ProfileInfo.Task task = info.allTasksById.get(0);
    assertEquals(1, task.id);
    assertEquals(ProfilerTask.ACTION, task.type);
    assertEquals("action task", task.getDescription());

    task = info.allTasksById.get(1);
    assertEquals(2, task.id);
    assertEquals(ProfilerTask.TEST, task.type);
    assertEquals("event", task.getDescription());
  }

  @Test
  public void testProfiler() throws Exception {
    Path cacheFile = cacheDir.getRelative("profile1.dat");
    profiler.start(ProfiledTaskKinds.ALL, cacheFile.getOutputStream(), "basic test", false,
        BlazeClock.instance(), BlazeClock.instance().nanoTime());
    profiler.logSimpleTask(getTestClock().nanoTime(),
                           ProfilerTask.PHASE, "profiler start");
    profiler.startTask(ProfilerTask.ACTION, "complex task");
    profiler.logEvent(ProfilerTask.PHASE, "event1");
    profiler.startTask(ProfilerTask.ACTION_CHECK, "complex subtask");
    // next task takes less than 10 ms and should be only aggregated
    profiler.logSimpleTask(getTestClock().nanoTime(),
                           ProfilerTask.VFS_STAT, "stat1");
    long startTime = getTestClock().nanoTime();
    Thread.sleep(20);
    // this one will take at least 20 ms and should be present
    profiler.logSimpleTask(startTime, ProfilerTask.VFS_STAT, "stat2");
    profiler.completeTask(ProfilerTask.ACTION_CHECK);
    profiler.completeTask(ProfilerTask.ACTION);
    profiler.stop();
    // all other calls to profiler should be ignored
    profiler.logEvent(ProfilerTask.PHASE, "should be ignored");
    // normally this would cause an exception but it is ignored since profiler
    // is disabled
    profiler.completeTask(ProfilerTask.ACTION_EXECUTE);

    ProfileInfo info = ProfileInfo.loadProfile(cacheFile);
    info.calculateStats();
    assertThat(info.allTasksById).hasSize(6); // only 5 tasks + finalization should be recorded

    ProfileInfo.Task task = info.allTasksById.get(0);
    assertTrue(task.stats.isEmpty());

    task = info.allTasksById.get(1);
    int count = 0;
    for (ProfileInfo.AggregateAttr attr : task.getStatAttrArray()) {
      if (attr != null) {
        count++;
      }
    }
    assertEquals(2, count); // only children are GENERIC and ACTION_CHECK
    assertEquals(task.aggregatedStats.toArray().length, ProfilerTask.TASK_COUNT);
    assertEquals(2, task.aggregatedStats.getAttr(ProfilerTask.VFS_STAT).count);

    task = info.allTasksById.get(2);
    assertThat(task.durationNanos).isEqualTo(0);

    task = info.allTasksById.get(3);
    assertEquals(2, task.stats.getAttr(ProfilerTask.VFS_STAT).count);
    assertEquals(1, task.subtasks.length);
    assertEquals("stat2", task.subtasks[0].getDescription());
    // assert that startTime grows with id
    long time = -1;
    for (ProfileInfo.Task t : info.allTasksById) {
      assertTrue(t.startTime >= time);
      time = t.startTime;
    }
  }

  @Test
  public void testProfilerRecordingAllEvents() throws Exception {
    Path cacheFile = cacheDir.getRelative("profile1.dat");
    profiler.start(ProfiledTaskKinds.ALL, cacheFile.getOutputStream(), "basic test", true,
        BlazeClock.instance(), BlazeClock.instance().nanoTime());
    profiler.startTask(ProfilerTask.ACTION, "action task");
    // Next task takes less than 10 ms but should be recorded anyway.
    profiler.logSimpleTask(getTestClock().nanoTime(), ProfilerTask.VFS_STAT, "stat1");
    profiler.completeTask(ProfilerTask.ACTION);
    profiler.stop();
    ProfileInfo info = ProfileInfo.loadProfile(cacheFile);
    info.calculateStats();
    assertThat(info.allTasksById).hasSize(3); // 2 tasks + finalization should be recorded

    ProfileInfo.Task task = info.allTasksById.get(1);
    assertEquals(ProfilerTask.VFS_STAT, task.type);

    // Check that task would have been dropped if profiler was not configured to record everything.
    assertThat(task.durationNanos).isLessThan(ProfilerTask.VFS_STAT.minDuration);
  }

  @Test
  public void testProfilerRecordingOnlySlowestEvents() throws Exception {
    Path profileData = cacheDir.getRelative("foo");

    profiler.start(ProfiledTaskKinds.SLOWEST, profileData.getOutputStream(), "test", true,
        BlazeClock.instance(), BlazeClock.instance().nanoTime());
    profiler.logSimpleTask(10000, 20000, ProfilerTask.VFS_STAT, "stat");
    profiler.logSimpleTask(20000, 30000, ProfilerTask.REMOTE_EXECUTION, "remote execution");

    assertTrue(profiler.isProfiling(ProfilerTask.VFS_STAT));
    assertFalse(profiler.isProfiling(ProfilerTask.REMOTE_EXECUTION));

    profiler.stop();

    ProfileInfo info = ProfileInfo.loadProfile(profileData);
    info.calculateStats();
    assertThat(info.allTasksById).hasSize(1); // only VFS_STAT task should be recorded

    ProfileInfo.Task task = info.allTasksById.get(0);
    assertEquals(ProfilerTask.VFS_STAT, task.type);
  }

  @Test
  public void testProfilerRecordsNothing() throws Exception {
    Path profileData = cacheDir.getRelative("foo");

    profiler.start(ProfiledTaskKinds.NONE, profileData.getOutputStream(), "test", true,
        BlazeClock.instance(), BlazeClock.instance().nanoTime());
    profiler.logSimpleTask(10000, 20000, ProfilerTask.VFS_STAT, "stat");

    assertTrue(ProfilerTask.VFS_STAT.collectsSlowestInstances());
    assertFalse(profiler.isProfiling(ProfilerTask.VFS_STAT));

    profiler.stop();

    ProfileInfo info = ProfileInfo.loadProfile(profileData);
    info.calculateStats();
    assertThat(info.allTasksById).isEmpty();
  }

  @Test
  public void testInconsistentCompleteTask() throws Exception {
    Path cacheFile = cacheDir.getRelative("profile2.dat");
    profiler.start(ProfiledTaskKinds.ALL, cacheFile.getOutputStream(),
        "task stack inconsistency test", false,
        BlazeClock.instance(), BlazeClock.instance().nanoTime());
    profiler.startTask(ProfilerTask.PHASE, "some task");
    try {
      profiler.completeTask(ProfilerTask.ACTION);
      fail();
    } catch (IllegalStateException e) {
      // this is expected
    }
    profiler.stop();
  }

  @Test
  public void testConcurrentProfiling() throws Exception {
    Path cacheFile = cacheDir.getRelative("profile3.dat");
    profiler.start(ProfiledTaskKinds.ALL, cacheFile.getOutputStream(), "concurrent test", false,
        BlazeClock.instance(), BlazeClock.instance().nanoTime());

    long id = Thread.currentThread().getId();
    Thread thread1 = new Thread() {
      @Override public void run() {
        for (int i = 0; i < 10000; i++) {
          Profiler.instance().logEvent(ProfilerTask.TEST, "thread1");
        }
      }
    };
    long id1 = thread1.getId();
    Thread thread2 = new Thread() {
      @Override public void run() {
        for (int i = 0; i < 10000; i++) {
          Profiler.instance().logEvent(ProfilerTask.TEST, "thread2");
        }
      }
    };
    long id2 = thread2.getId();

    profiler.startTask(ProfilerTask.PHASE, "main task");
    profiler.logEvent(ProfilerTask.TEST, "starting threads");
    thread1.start();
    thread2.start();
    thread2.join();
    thread1.join();
    profiler.logEvent(ProfilerTask.TEST, "joined");
    profiler.completeTask(ProfilerTask.PHASE);
    profiler.stop();

    ProfileInfo info = ProfileInfo.loadProfile(cacheFile);
    info.calculateStats();
    info.analyzeRelationships();
    assertEquals(4 + 10000 + 10000, info.allTasksById.size()); // total number of tasks
    assertEquals(3, info.tasksByThread.size()); // total number of threads
    // while main thread had 3 tasks, 2 of them were nested, so tasksByThread
    // would contain only one "main task" task
    assertEquals(2, info.tasksByThread.get(id).length);
    ProfileInfo.Task mainTask = info.tasksByThread.get(id)[0];
    assertEquals("main task", mainTask.getDescription());
    assertEquals(2, mainTask.subtasks.length);
    // other threads had 10000 independent recorded tasks each
    assertEquals(10000, info.tasksByThread.get(id1).length);
    assertEquals(10000, info.tasksByThread.get(id2).length);
    int startId = mainTask.subtasks[0].id; // id of "starting threads"
    int endId = mainTask.subtasks[1].id; // id of "joining"
    assertTrue(startId < info.tasksByThread.get(id1)[0].id);
    assertTrue(startId < info.tasksByThread.get(id2)[0].id);
    assertTrue(endId > info.tasksByThread.get(id1)[9999].id);
    assertTrue(endId > info.tasksByThread.get(id2)[9999].id);
  }

  @Test
  public void testPhaseTasks() throws Exception {
    Path cacheFile = cacheDir.getRelative("profile4.dat");
    profiler.start(ProfiledTaskKinds.ALL, cacheFile.getOutputStream(), "phase test", false,
        BlazeClock.instance(), BlazeClock.instance().nanoTime());
    Thread thread1 = new Thread() {
      @Override public void run() {
        for (int i = 0; i < 100; i++) {
          Profiler.instance().logEvent(ProfilerTask.TEST, "thread1");
        }
      }
    };
    profiler.markPhase(ProfilePhase.INIT); // Empty phase.
    profiler.markPhase(ProfilePhase.LOAD);
    thread1.start();
    thread1.join();
    Thread.sleep(1);
    profiler.markPhase(ProfilePhase.ANALYZE);
    Thread thread2 = new Thread() {
      @Override public void run() {
        profiler.startTask(ProfilerTask.TEST, "complex task");
        for (int i = 0; i < 100; i++) {
          Profiler.instance().logEvent(ProfilerTask.TEST, "thread2a");
        }
        profiler.completeTask(ProfilerTask.TEST);
        profiler.markPhase(ProfilePhase.EXECUTE);
        for (int i = 0; i < 100; i++) {
          Profiler.instance().logEvent(ProfilerTask.TEST, "thread2b");
        }
      }
    };
    thread2.start();
    thread2.join();
    profiler.logEvent(ProfilerTask.TEST, "last task");
    Thread.sleep(1);
    profiler.stop();

    ProfileInfo info = ProfileInfo.loadProfile(cacheFile);
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
    Path cacheFile = cacheDir.getRelative("profile5.dat");
    profiler.start(ProfiledTaskKinds.ALL, cacheFile.getOutputStream(), "phase test", false,
        BlazeClock.instance(), BlazeClock.instance().nanoTime());
    for (int i = 0; i < 100; i++) {
      profiler.startTask(ProfilerTask.TEST, "outer task " + i);
      profiler.logEvent(ProfilerTask.TEST, "inner task " + i);
      profiler.completeTask(ProfilerTask.TEST);
    }
    profiler.stop();

    ProfileInfo info = ProfileInfo.loadProfile(cacheFile);
    info.calculateStats();
    assertFalse(info.isCorruptedOrIncomplete());

    Path corruptedFile = cacheDir.getRelative("profile5bad.dat");
    FileSystemUtils.writeContent(
        corruptedFile, Arrays.copyOf(FileSystemUtils.readContent(cacheFile), 2000));
    info = ProfileInfo.loadProfile(corruptedFile);
    info.calculateStats();
    assertTrue(info.isCorruptedOrIncomplete());
    // Since root tasks will appear after nested tasks in the profile file and
    // we have exactly one nested task for each root task, then following will be
    // always true for our corrupted file:
    // 0 <= number_of_all_tasks - 2*number_of_root_tasks <= 1
    assertEquals(info.rootTasksById.size(), info.allTasksById.size() / 2);
  }

  @Test
  public void testUnsupportedProfilerRecord() throws Exception {
    Path dataFile = cacheDir.getRelative("profile5.dat");
    profiler.start(ProfiledTaskKinds.ALL, dataFile.getOutputStream(), "phase test", false,
        BlazeClock.instance(), BlazeClock.instance().nanoTime());
    profiler.startTask(ProfilerTask.TEST, "outer task");
    profiler.logEvent(ProfilerTask.EXCEPTION, "inner task");
    profiler.completeTask(ProfilerTask.TEST);
    profiler.startTask(ProfilerTask.SCANNER, "outer task 2");
    profiler.logSimpleTask(Profiler.nanoTimeMaybe(), ProfilerTask.TEST, "inner task 2");
    profiler.completeTask(ProfilerTask.SCANNER);
    profiler.stop();

    // Validate our test profile.
    ProfileInfo info = ProfileInfo.loadProfile(dataFile);
    info.calculateStats();
    assertFalse(info.isCorruptedOrIncomplete());
    assertEquals(2, info.getStatsForType(ProfilerTask.TEST, info.rootTasksById).count);
    assertEquals(0, info.getStatsForType(ProfilerTask.UNKNOWN, info.rootTasksById).count);

    // Now replace "TEST" type with something unsupported - e.g. "XXXX".
    InputStream in = new InflaterInputStream(dataFile.getInputStream(), new Inflater(false), 65536);
    byte[] buffer = new byte[60000];
    int len = in.read(buffer);
    in.close();
    assertTrue(len < buffer.length); // Validate that file was completely decoded.
    String content = new String(buffer, ISO_8859_1);
    int infoIndex = content.indexOf("TEST");
    assertTrue(infoIndex > 0);
    content = content.substring(0, infoIndex) + "XXXX" + content.substring(infoIndex + 4);
    OutputStream out = new DeflaterOutputStream(dataFile.getOutputStream(),
        new Deflater(Deflater.BEST_SPEED, false), 65536);
    out.write(content.getBytes(ISO_8859_1));
    out.close();

    // Validate that XXXX records were classified as UNKNOWN.
    info = ProfileInfo.loadProfile(dataFile);
    info.calculateStats();
    assertFalse(info.isCorruptedOrIncomplete());
    assertEquals(0, info.getStatsForType(ProfilerTask.TEST, info.rootTasksById).count);
    assertEquals(1, info.getStatsForType(ProfilerTask.SCANNER, info.rootTasksById).count);
    assertEquals(1, info.getStatsForType(ProfilerTask.EXCEPTION, info.rootTasksById).count);
    assertEquals(2, info.getStatsForType(ProfilerTask.UNKNOWN, info.rootTasksById).count);
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
    Path cacheFile = cacheDir.getRelative("profile1.dat");
    profiler.start(ProfiledTaskKinds.ALL, cacheFile.getOutputStream(),
        "testResilenceToNonDecreasingNanoTimes", false, badClock, initialNanoTime);
    profiler.logSimpleTask(badClock.nanoTime(), ProfilerTask.TEST, "some task");
    profiler.stop();
  }

  private Clock getTestClock() {
    return BlazeClock.instance();
  }

}
