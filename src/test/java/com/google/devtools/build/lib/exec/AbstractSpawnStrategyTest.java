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
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnExecutedEvent;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.exec.SpawnCache.CacheHandle;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Spawn.Code;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

/** Tests for {@link BlazeExecutor}. */
@RunWith(JUnit4.class)
public class AbstractSpawnStrategyTest {
  private static final FailureDetail NON_ZERO_EXIT_DETAILS =
      FailureDetail.newBuilder()
          .setSpawn(FailureDetails.Spawn.newBuilder().setCode(Code.NON_ZERO_EXIT))
          .build();

  private static class TestedSpawnStrategy extends AbstractSpawnStrategy {
    TestedSpawnStrategy(SpawnRunner spawnRunner) {
      super(spawnRunner, new ExecutionOptions());
    }
  }

  private static final Spawn SIMPLE_SPAWN =
      new SpawnBuilder("/bin/echo", "Hi!").withEnvironment("VARIABLE", "value").build();

  private final FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
  private final Path execRoot = fs.getPath("/execroot");
  @Mock private SpawnRunner spawnRunner;
  @Mock private ActionExecutionContext actionExecutionContext;
  private StoredEventHandler eventHandler;
  private final ManualClock clock = new ManualClock();

  @Before
  public final void setUp() throws Exception {
    MockitoAnnotations.initMocks(this);
    eventHandler = new StoredEventHandler();
    when(actionExecutionContext.getEventHandler()).thenReturn(eventHandler);
    when(actionExecutionContext.getClock()).thenReturn(clock);
  }

  @Test
  public void testZeroExit() throws Exception {
    when(actionExecutionContext.getContext(eq(SpawnCache.class))).thenReturn(SpawnCache.NO_CACHE);
    when(actionExecutionContext.getExecRoot()).thenReturn(execRoot);
    SpawnResult spawnResult =
        new SpawnResult.Builder().setStatus(Status.SUCCESS).setRunnerName("test").build();
    when(spawnRunner.exec(any(Spawn.class), any(SpawnExecutionContext.class)))
        .thenReturn(spawnResult);

    List<SpawnResult> spawnResults =
        new TestedSpawnStrategy(spawnRunner).exec(SIMPLE_SPAWN, actionExecutionContext);

    assertThat(spawnResults).containsExactly(spawnResult);

    // Must only be called exactly once.
    verify(spawnRunner).exec(any(Spawn.class), any(SpawnExecutionContext.class));
  }

  @Test
  public void testEventPosting() throws Exception {
    when(actionExecutionContext.getContext(eq(SpawnCache.class))).thenReturn(SpawnCache.NO_CACHE);
    when(actionExecutionContext.getExecRoot()).thenReturn(execRoot);
    SpawnResult spawnResult =
        new SpawnResult.Builder().setStatus(Status.SUCCESS).setRunnerName("test").build();
    Instant beforeTime = Instant.ofEpochMilli(clock.currentTimeMillis());
    doAnswer(
            invocation -> {
              clock.advanceMillis(1);
              return spawnResult;
            })
        .when(spawnRunner)
        .exec(any(Spawn.class), any(SpawnExecutionContext.class));

    ImmutableList<SpawnResult> spawnResults =
        new TestedSpawnStrategy(spawnRunner).exec(SIMPLE_SPAWN, actionExecutionContext);

    assertThat(spawnResults).containsExactly(spawnResult);
    // Must only be called exactly once.
    verify(spawnRunner).exec(any(Spawn.class), any(SpawnExecutionContext.class));
    assertThat(eventHandler.getPosts()).hasSize(1);
    SpawnExecutedEvent event = (SpawnExecutedEvent) eventHandler.getPosts().get(0);
    assertThat(event.getStartTimeInstant()).isEqualTo(beforeTime);
    assertThat(event.getSpawnResult()).isEqualTo(spawnResult);
    assertThat(event.getExitCode()).isEqualTo(0);
  }

  @Test
  public void testNonZeroExit() throws Exception {
    when(actionExecutionContext.getContext(eq(SpawnCache.class))).thenReturn(SpawnCache.NO_CACHE);
    when(actionExecutionContext.getExecRoot()).thenReturn(execRoot);
    SpawnResult result =
        new SpawnResult.Builder()
            .setStatus(Status.NON_ZERO_EXIT)
            .setExitCode(1)
            .setFailureDetail(NON_ZERO_EXIT_DETAILS)
            .setRunnerName("test")
            .build();
    when(spawnRunner.exec(any(Spawn.class), any(SpawnExecutionContext.class))).thenReturn(result);

    SpawnExecException e =
        assertThrows(
            SpawnExecException.class,
            () ->
                // Ignoring the List<SpawnResult> return value.
                new TestedSpawnStrategy(spawnRunner).exec(SIMPLE_SPAWN, actionExecutionContext));
    assertThat(e.getSpawnResult()).isSameInstanceAs(result);
    // Must only be called exactly once.
    verify(spawnRunner).exec(any(Spawn.class), any(SpawnExecutionContext.class));
  }

  @Test
  public void testCacheHit() throws Exception {
    SpawnCache cache = mock(SpawnCache.class);
    SpawnResult spawnResult =
        new SpawnResult.Builder().setStatus(Status.SUCCESS).setRunnerName("test").build();
    when(cache.lookup(any(Spawn.class), any(SpawnExecutionContext.class)))
        .thenReturn(SpawnCache.success(spawnResult));
    when(actionExecutionContext.getContext(eq(SpawnCache.class))).thenReturn(cache);
    when(actionExecutionContext.getExecRoot()).thenReturn(execRoot);

    List<SpawnResult> spawnResults =
        new TestedSpawnStrategy(spawnRunner).exec(SIMPLE_SPAWN, actionExecutionContext);
    assertThat(spawnResults).containsExactly(spawnResult);
    verify(spawnRunner, never()).exec(any(Spawn.class), any(SpawnExecutionContext.class));
  }

  @Test
  public void testCacheMiss() throws Exception {
    SpawnCache cache = mock(SpawnCache.class);
    CacheHandle entry = mock(CacheHandle.class);
    when(cache.lookup(any(Spawn.class), any(SpawnExecutionContext.class))).thenReturn(entry);
    when(entry.hasResult()).thenReturn(false);
    when(entry.willStore()).thenReturn(true);

    when(actionExecutionContext.getContext(eq(SpawnCache.class))).thenReturn(cache);
    when(actionExecutionContext.getExecRoot()).thenReturn(execRoot);
    SpawnResult spawnResult =
        new SpawnResult.Builder().setStatus(Status.SUCCESS).setRunnerName("test").build();
    when(spawnRunner.exec(any(Spawn.class), any(SpawnExecutionContext.class)))
        .thenReturn(spawnResult);

    List<SpawnResult> spawnResults =
        new TestedSpawnStrategy(spawnRunner).exec(SIMPLE_SPAWN, actionExecutionContext);

    assertThat(spawnResults).containsExactly(spawnResult);

    // Must only be called exactly once.
    verify(spawnRunner).exec(any(Spawn.class), any(SpawnExecutionContext.class));
    verify(entry).store(eq(spawnResult));
  }

  @Test
  public void testExec_whenLocalCaches_usesNoCache() throws Exception {
    when(spawnRunner.handlesCaching()).thenReturn(true);

    SpawnCache cache = mock(SpawnCache.class);

    when(actionExecutionContext.getContext(eq(SpawnCache.class))).thenReturn(cache);
    when(actionExecutionContext.getExecRoot()).thenReturn(execRoot);
    SpawnResult spawnResult =
        new SpawnResult.Builder().setStatus(Status.SUCCESS).setRunnerName("test").build();
    when(spawnRunner.exec(any(Spawn.class), any(SpawnExecutionContext.class)))
        .thenReturn(spawnResult);

    List<SpawnResult> spawnResults =
        new TestedSpawnStrategy(spawnRunner).exec(SIMPLE_SPAWN, actionExecutionContext);

    assertThat(spawnResults).containsExactly(spawnResult);

    // Must only be called exactly once.
    verify(spawnRunner).exec(any(Spawn.class), any(SpawnExecutionContext.class));
    verifyNoInteractions(cache);
  }

  @Test
  public void testExec_usefulCacheInDynamicExecution() throws Exception {
    when(spawnRunner.handlesCaching()).thenReturn(false);

    SpawnCache cache = mock(SpawnCache.class);
    when(cache.usefulInDynamicExecution()).thenReturn(true);
    CacheHandle entry = mock(CacheHandle.class);
    when(cache.lookup(any(Spawn.class), any(SpawnExecutionContext.class))).thenReturn(entry);
    when(entry.hasResult()).thenReturn(false);
    when(entry.willStore()).thenReturn(true);

    when(actionExecutionContext.getContext(eq(SpawnCache.class))).thenReturn(cache);
    when(actionExecutionContext.getExecRoot()).thenReturn(execRoot);
    SpawnResult spawnResult =
        new SpawnResult.Builder().setStatus(Status.SUCCESS).setRunnerName("test").build();
    when(spawnRunner.exec(any(Spawn.class), any(SpawnExecutionContext.class)))
        .thenReturn(spawnResult);

    List<SpawnResult> spawnResults =
        new TestedSpawnStrategy(spawnRunner)
            .exec(SIMPLE_SPAWN, actionExecutionContext, (exitCode, errorMessage, outErr) -> {});

    assertThat(spawnResults).containsExactly(spawnResult);

    // Must only be called exactly once.
    verify(spawnRunner).exec(any(Spawn.class), any(SpawnExecutionContext.class));
    verify(entry).store(eq(spawnResult));
  }

  @Test
  public void testExec_nonUsefulCacheInDynamicExecution() throws Exception {
    when(spawnRunner.handlesCaching()).thenReturn(false);

    SpawnCache cache = mock(SpawnCache.class);
    when(cache.usefulInDynamicExecution()).thenReturn(false);

    when(actionExecutionContext.getContext(eq(SpawnCache.class))).thenReturn(cache);
    when(actionExecutionContext.getExecRoot()).thenReturn(execRoot);
    SpawnResult spawnResult =
        new SpawnResult.Builder().setStatus(Status.SUCCESS).setRunnerName("test").build();
    when(spawnRunner.exec(any(Spawn.class), any(SpawnExecutionContext.class)))
        .thenReturn(spawnResult);

    List<SpawnResult> spawnResults =
        new TestedSpawnStrategy(spawnRunner)
            .exec(SIMPLE_SPAWN, actionExecutionContext, (exitCode, errorMessage, outErr) -> {});

    assertThat(spawnResults).containsExactly(spawnResult);

    // Must only be called exactly once.
    verify(spawnRunner).exec(any(Spawn.class), any(SpawnExecutionContext.class));
    verify(cache).usefulInDynamicExecution();
    verifyNoMoreInteractions(cache);
  }

  @Test
  public void testCacheMissWithNonZeroExit() throws Exception {
    SpawnCache cache = mock(SpawnCache.class);
    CacheHandle entry = mock(CacheHandle.class);
    when(cache.lookup(any(Spawn.class), any(SpawnExecutionContext.class))).thenReturn(entry);
    when(entry.hasResult()).thenReturn(false);
    when(entry.willStore()).thenReturn(true);

    when(actionExecutionContext.getContext(eq(SpawnCache.class))).thenReturn(cache);
    when(actionExecutionContext.getExecRoot()).thenReturn(execRoot);
    SpawnResult result =
        new SpawnResult.Builder()
            .setStatus(Status.NON_ZERO_EXIT)
            .setExitCode(1)
            .setFailureDetail(NON_ZERO_EXIT_DETAILS)
            .setRunnerName("test")
            .build();
    when(spawnRunner.exec(any(Spawn.class), any(SpawnExecutionContext.class))).thenReturn(result);

    SpawnExecException e =
        assertThrows(
            SpawnExecException.class,
            () ->
                // Ignoring the List<SpawnResult> return value.
                new TestedSpawnStrategy(spawnRunner).exec(SIMPLE_SPAWN, actionExecutionContext));
    assertThat(e.getSpawnResult()).isSameInstanceAs(result);
    // Must only be called exactly once.
    verify(spawnRunner).exec(any(Spawn.class), any(SpawnExecutionContext.class));
    verify(entry).store(eq(result));
  }

  @Test
  public void testExec_callsLogSpawn() throws Exception {
    FileSystem actionFs = mock(FileSystem.class);
    InputMetadataProvider inputMetadataProvider = mock(InputMetadataProvider.class);
    SpawnLogContext spawnLogContext = mock(SpawnLogContext.class);

    SpawnResult spawnResult =
        new SpawnResult.Builder().setStatus(Status.SUCCESS).setRunnerName("test").build();

    when(actionExecutionContext.getContext(eq(SpawnCache.class))).thenReturn(SpawnCache.NO_CACHE);
    when(actionExecutionContext.getContext(eq(SpawnLogContext.class))).thenReturn(spawnLogContext);
    when(actionExecutionContext.getExecRoot()).thenReturn(execRoot);
    when(actionExecutionContext.getActionFileSystem()).thenReturn(actionFs);
    when(actionExecutionContext.getInputMetadataProvider()).thenReturn(inputMetadataProvider);

    when(spawnRunner.exec(any(Spawn.class), any(SpawnExecutionContext.class)))
        .thenReturn(spawnResult);

    ImmutableList<SpawnResult> spawnResults =
        new TestedSpawnStrategy(spawnRunner).exec(SIMPLE_SPAWN, actionExecutionContext);
    assertThat(spawnResults).containsExactly(spawnResult);

    verify(spawnLogContext)
        .logSpawn(
            eq(SIMPLE_SPAWN),
            eq(inputMetadataProvider),
            any(),
            eq(actionFs),
            eq(Duration.ZERO),
            eq(spawnResult));
  }
}
