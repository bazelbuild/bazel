// Copyright 2017 The Bazel Authors. All Rights Reserved.
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
package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;
import static org.mockito.Matchers.any;
import static org.mockito.Matchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.exec.SpawnCache.CacheHandle;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionPolicy;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.Collection;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

/** Tests for {@link BlazeExecutor}. */
@RunWith(JUnit4.class)
@TestSpec(size = Suite.SMALL_TESTS)
public class AbstractSpawnStrategyTest {
  private static class TestedSpawnStrategy extends AbstractSpawnStrategy {
    public TestedSpawnStrategy(SpawnRunner spawnRunner) {
      super(spawnRunner);
    }
  }

  private static final Spawn SIMPLE_SPAWN =
      new SpawnBuilder("/bin/echo", "Hi!").withEnvironment("VARIABLE", "value").build();

  private final FileSystem fs = new InMemoryFileSystem();
  @Mock private SpawnRunner spawnRunner;
  @Mock private ActionExecutionContext actionExecutionContext;

  @Before
  public final void setUp() throws Exception {
    MockitoAnnotations.initMocks(this);
  }

  @Test
  public void testZeroExit() throws Exception {
    when(actionExecutionContext.getContext(eq(SpawnCache.class))).thenReturn(SpawnCache.NO_CACHE);
    when(actionExecutionContext.getExecRoot()).thenReturn(fs.getPath("/execroot"));
    SpawnResult spawnResult = new SpawnResult.Builder().setStatus(Status.SUCCESS).build();
    when(spawnRunner.exec(any(Spawn.class), any(SpawnExecutionPolicy.class)))
        .thenReturn(spawnResult);

    Set<SpawnResult> spawnResults = new TestedSpawnStrategy(spawnRunner)
        .exec(SIMPLE_SPAWN, actionExecutionContext);

    assertThat(spawnResults).containsExactly(spawnResult);

    // Must only be called exactly once.
    verify(spawnRunner).exec(any(Spawn.class), any(SpawnExecutionPolicy.class));
  }

  @Test
  public void testNonZeroExit() throws Exception {
    when(actionExecutionContext.getContext(eq(SpawnCache.class))).thenReturn(SpawnCache.NO_CACHE);
    when(actionExecutionContext.getExecRoot()).thenReturn(fs.getPath("/execroot"));
    SpawnResult result = new SpawnResult.Builder().setStatus(Status.SUCCESS).setExitCode(1).build();
    when(spawnRunner.exec(any(Spawn.class), any(SpawnExecutionPolicy.class)))
        .thenReturn(result);

    try {
      // Ignoring the Set<SpawnResult> return value.
      new TestedSpawnStrategy(spawnRunner).exec(SIMPLE_SPAWN, actionExecutionContext);
      fail("Expected SpawnExecException");
    } catch (SpawnExecException e) {
      assertThat(e.getSpawnResult()).isSameAs(result);
    }
    // Must only be called exactly once.
    verify(spawnRunner).exec(any(Spawn.class), any(SpawnExecutionPolicy.class));
  }

  @Test
  public void testCacheHit() throws Exception {
    SpawnCache cache = mock(SpawnCache.class);
    SpawnResult spawnResult = new SpawnResult.Builder().setStatus(Status.SUCCESS).build();
    when(cache.lookup(any(Spawn.class), any(SpawnExecutionPolicy.class)))
        .thenReturn(SpawnCache.success(spawnResult));
    when(actionExecutionContext.getContext(eq(SpawnCache.class))).thenReturn(cache);
    when(actionExecutionContext.getExecRoot()).thenReturn(fs.getPath("/execroot"));

    Set<SpawnResult> spawnResults = new TestedSpawnStrategy(spawnRunner)
        .exec(SIMPLE_SPAWN, actionExecutionContext);
    assertThat(spawnResults).containsExactly(spawnResult);
    verify(spawnRunner, never()).exec(any(Spawn.class), any(SpawnExecutionPolicy.class));
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testCacheMiss() throws Exception {
    SpawnCache cache = mock(SpawnCache.class);
    CacheHandle entry = mock(CacheHandle.class);
    when(cache.lookup(any(Spawn.class), any(SpawnExecutionPolicy.class))).thenReturn(entry);
    when(entry.hasResult()).thenReturn(false);
    when(entry.willStore()).thenReturn(true);

    when(actionExecutionContext.getContext(eq(SpawnCache.class))).thenReturn(cache);
    when(actionExecutionContext.getExecRoot()).thenReturn(fs.getPath("/execroot"));
    SpawnResult spawnResult = new SpawnResult.Builder().setStatus(Status.SUCCESS).build();
    when(spawnRunner.exec(any(Spawn.class), any(SpawnExecutionPolicy.class)))
        .thenReturn(spawnResult);

    Set<SpawnResult> spawnResults = new TestedSpawnStrategy(spawnRunner)
        .exec(SIMPLE_SPAWN, actionExecutionContext);

    assertThat(spawnResults).containsExactly(spawnResult);

    // Must only be called exactly once.
    verify(spawnRunner).exec(any(Spawn.class), any(SpawnExecutionPolicy.class));
    verify(entry).store(eq(spawnResult), any(Collection.class));
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testCacheMissWithNonZeroExit() throws Exception {
    SpawnCache cache = mock(SpawnCache.class);
    CacheHandle entry = mock(CacheHandle.class);
    when(cache.lookup(any(Spawn.class), any(SpawnExecutionPolicy.class))).thenReturn(entry);
    when(entry.hasResult()).thenReturn(false);
    when(entry.willStore()).thenReturn(true);

    when(actionExecutionContext.getContext(eq(SpawnCache.class))).thenReturn(cache);
    when(actionExecutionContext.getExecRoot()).thenReturn(fs.getPath("/execroot"));
    SpawnResult result = new SpawnResult.Builder().setStatus(Status.SUCCESS).setExitCode(1).build();
    when(spawnRunner.exec(any(Spawn.class), any(SpawnExecutionPolicy.class))).thenReturn(result);

    try {
      // Ignoring the Set<SpawnResult> return value.
      new TestedSpawnStrategy(spawnRunner).exec(SIMPLE_SPAWN, actionExecutionContext);
      fail("Expected SpawnExecException");
    } catch (SpawnExecException e) {
      assertThat(e.getSpawnResult()).isSameAs(result);
    }
    // Must only be called exactly once.
    verify(spawnRunner).exec(any(Spawn.class), any(SpawnExecutionPolicy.class));
    verify(entry).store(eq(result), any(Collection.class));
  }

  @Test
  public void testTagNoCache() throws Exception {
    SpawnCache cache = mock(SpawnCache.class);
    when(actionExecutionContext.getContext(eq(SpawnCache.class))).thenReturn(cache);
    when(actionExecutionContext.getExecRoot()).thenReturn(fs.getPath("/execroot"));
    when(spawnRunner.exec(any(Spawn.class), any(SpawnExecutionPolicy.class)))
        .thenReturn(new SpawnResult.Builder().setStatus(Status.SUCCESS).build());

    Spawn uncacheableSpawn =
        new SpawnBuilder("/bin/echo", "Hi").withExecutionInfo("no-cache", "").build();

    new TestedSpawnStrategy(spawnRunner).exec(uncacheableSpawn, actionExecutionContext);

    // Must only be called exactly once.
    verify(spawnRunner).exec(any(Spawn.class), any(SpawnExecutionPolicy.class));
    // Must not be called.
    verify(cache, never()).lookup(any(Spawn.class), any(SpawnExecutionPolicy.class));
  }
}
