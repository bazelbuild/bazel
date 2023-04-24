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
import static org.mockito.ArgumentMatchers.argThat;
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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnExecutedEvent;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.exec.Protos.Digest;
import com.google.devtools.build.lib.exec.Protos.EnvironmentVariable;
import com.google.devtools.build.lib.exec.Protos.File;
import com.google.devtools.build.lib.exec.Protos.Platform;
import com.google.devtools.build.lib.exec.Protos.SpawnExec;
import com.google.devtools.build.lib.exec.SpawnCache.CacheHandle;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Spawn.Code;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.io.MessageOutputStream;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import com.google.protobuf.Duration;
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
    public TestedSpawnStrategy(Path execRoot, SpawnRunner spawnRunner) {
      super(execRoot, spawnRunner, new ExecutionOptions());
    }
  }

  private static final Spawn SIMPLE_SPAWN =
      new SpawnBuilder("/bin/echo", "Hi!").withEnvironment("VARIABLE", "value").build();

  private final FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
  private final Path execRoot = fs.getPath("/execroot");
  private Scratch scratch;
  private ArtifactRoot rootDir;
  @Mock private SpawnRunner spawnRunner;
  @Mock private ActionExecutionContext actionExecutionContext;
  @Mock private MessageOutputStream messageOutput;
  private StoredEventHandler eventHandler;
  private final ManualClock clock = new ManualClock();

  @Before
  public final void setUp() throws Exception {
    MockitoAnnotations.initMocks(this);
    scratch = new Scratch(fs);
    rootDir = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/execroot")));
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
        new TestedSpawnStrategy(execRoot, spawnRunner).exec(SIMPLE_SPAWN, actionExecutionContext);

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
        new TestedSpawnStrategy(execRoot, spawnRunner).exec(SIMPLE_SPAWN, actionExecutionContext);

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
                new TestedSpawnStrategy(execRoot, spawnRunner)
                    .exec(SIMPLE_SPAWN, actionExecutionContext));
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
        new TestedSpawnStrategy(execRoot, spawnRunner).exec(SIMPLE_SPAWN, actionExecutionContext);
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
        new TestedSpawnStrategy(execRoot, spawnRunner).exec(SIMPLE_SPAWN, actionExecutionContext);

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
        new TestedSpawnStrategy(execRoot, spawnRunner).exec(SIMPLE_SPAWN, actionExecutionContext);

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
        new TestedSpawnStrategy(execRoot, spawnRunner)
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
        new TestedSpawnStrategy(execRoot, spawnRunner)
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
                new TestedSpawnStrategy(execRoot, spawnRunner)
                    .exec(SIMPLE_SPAWN, actionExecutionContext));
    assertThat(e.getSpawnResult()).isSameInstanceAs(result);
    // Must only be called exactly once.
    verify(spawnRunner).exec(any(Spawn.class), any(SpawnExecutionContext.class));
    verify(entry).store(eq(result));
  }

  @Test
  public void testLogSpawn() throws Exception {
    setUpExecutionContext(/* executionOptions= */ null, /* remoteOptions= */ null);

    Artifact input = ActionsTestUtil.createArtifact(rootDir, scratch.file("/execroot/foo", "1"));
    scratch.file("/execroot/out1", "123");
    scratch.file("/execroot/out2", "123");
    Spawn spawn =
        new SpawnBuilder("/bin/echo", "Foo!")
            .withEnvironment("FOO", "v1")
            .withEnvironment("BAR", "v2")
            .withMnemonic("MyMnemonic")
            .withProgressMessage("my progress message")
            .withInput(input)
            .withOutputs("out2", "out1")
            .build();
    assertThrows(
        SpawnExecException.class,
        () -> new TestedSpawnStrategy(execRoot, spawnRunner).exec(spawn, actionExecutionContext));

    SpawnExec expectedSpawnLog =
        SpawnExec.newBuilder()
            .addCommandArgs("/bin/echo")
            .addCommandArgs("Foo!")
            .addEnvironmentVariables(
                EnvironmentVariable.newBuilder().setName("BAR").setValue("v2").build())
            .addEnvironmentVariables(
                EnvironmentVariable.newBuilder().setName("FOO").setValue("v1").build())
            .addInputs(
                File.newBuilder()
                    .setPath("foo")
                    .setDigest(
                        Digest.newBuilder()
                            .setHash(
                                "4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865")
                            .setSizeBytes(2)
                            .setHashFunctionName("SHA-256")
                            .build())
                    .build())
            .addListedOutputs("out1")
            .addListedOutputs("out2")
            .addActualOutputs(
                File.newBuilder()
                    .setPath("out1")
                    .setDigest(
                        Digest.newBuilder()
                            .setHash(
                                "181210f8f9c779c26da1d9b2075bde0127302ee0e3fca38c9a83f5b1dd8e5d3b")
                            .setSizeBytes(4)
                            .setHashFunctionName("SHA-256")
                            .build())
                    .build())
            .addActualOutputs(
                File.newBuilder()
                    .setPath("out2")
                    .setDigest(
                        Digest.newBuilder()
                            .setHash(
                                "181210f8f9c779c26da1d9b2075bde0127302ee0e3fca38c9a83f5b1dd8e5d3b")
                            .setSizeBytes(4)
                            .setHashFunctionName("SHA-256")
                            .build())
                    .build())
            .setStatus("NON_ZERO_EXIT")
            .setExitCode(23)
            .setRemotable(true)
            .setCacheable(true)
            .setRemoteCacheable(true)
            .setProgressMessage("my progress message")
            .setMnemonic("MyMnemonic")
            .setRunner("runner")
            .setWalltime(Duration.getDefaultInstance())
            .setTargetLabel("//dummy:label")
            .build();
    verify(messageOutput).write(expectedSpawnLog);
  }

  @Test
  public void testLogSpawn_noPlatform_noLoggedPlatform() throws Exception {
    setUpExecutionContext(/* executionOptions= */ null, /* remoteOptions= */ null);

    Spawn spawn = new SpawnBuilder("cmd").build();

    assertThrows(
        SpawnExecException.class,
        () -> new TestedSpawnStrategy(execRoot, spawnRunner).exec(spawn, actionExecutionContext));

    SpawnExec expected = defaultSpawnExecBuilder("cmd").build();
    verify(messageOutput).write(expected);
  }

  @Test
  public void testLogSpawn_defaultPlatform_getsLogged() throws Exception {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    remoteOptions.remoteDefaultPlatformProperties =
        String.join(
            "\n",
            "properties: {",
            " name: \"b\"",
            " value: \"2\"",
            "}",
            "properties: {",
            " name: \"a\"",
            " value: \"1\"",
            "}");

    setUpExecutionContext(/* executionOptions= */ null, remoteOptions);
    Spawn spawn = new SpawnBuilder("cmd").build();
    assertThrows(
        SpawnExecException.class,
        () -> new TestedSpawnStrategy(execRoot, spawnRunner).exec(spawn, actionExecutionContext));

    Platform platform =
        Platform.newBuilder()
            .addProperties(Platform.Property.newBuilder().setName("a").setValue("1"))
            .addProperties(Platform.Property.newBuilder().setName("b").setValue("2"))
            .build();
    SpawnExec expected = defaultSpawnExecBuilder("cmd").setPlatform(platform).build();
    verify(messageOutput).write(expected); // output will reflect default properties
  }

  @Test
  public void testLogSpawn_spawnMetrics() throws Exception {
    ExecutionOptions executionOptions = Options.getDefaults(ExecutionOptions.class);
    executionOptions.executionLogSpawnMetrics = true;

    setUpExecutionContext(executionOptions, /* remoteOptions= */ null);

    assertThrows(
        SpawnExecException.class,
        () ->
            new TestedSpawnStrategy(execRoot, spawnRunner)
                .exec(SIMPLE_SPAWN, actionExecutionContext));

    verify(messageOutput).write(argThat((m) -> ((SpawnExec) m).hasMetrics()));
  }

  @Test
  public void testLogSpawn_specifiedPlatform_overridesDefault() throws Exception {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    remoteOptions.remoteDefaultPlatformProperties =
        String.join(
            "\n",
            "properties: {",
            " name: \"b\"",
            " value: \"2\"",
            "}",
            "properties: {",
            " name: \"a\"",
            " value: \"1\"",
            "}");
    setUpExecutionContext(/* executionOptions= */ null, remoteOptions);

    PlatformInfo platformInfo =
        PlatformInfo.builder()
            .setRemoteExecutionProperties(
                String.join(
                    "\n",
                    "properties: {",
                    " name: \"new\"",
                    " value: \"val\"",
                    "}",
                    "properties: {",
                    " name: \"new2\"",
                    " value: \"val2\"",
                    "}"))
            .build();
    Spawn spawn = new SpawnBuilder("cmd").withPlatform(platformInfo).build();

    assertThrows(
        SpawnExecException.class,
        () -> new TestedSpawnStrategy(execRoot, spawnRunner).exec(spawn, actionExecutionContext));

    // If the platform is specified on the spawn, it is used even if a default if specified in the
    // options.
    Platform platform =
        Platform.newBuilder()
            .addProperties(Platform.Property.newBuilder().setName("new").setValue("val"))
            .addProperties(Platform.Property.newBuilder().setName("new2").setValue("val2"))
            .build();
    SpawnExec expected = defaultSpawnExecBuilder("cmd").setPlatform(platform).build();

    verify(messageOutput).write(expected); // output will reflect default properties
  }

  private void setUpExecutionContext(ExecutionOptions executionOptions, RemoteOptions remoteOptions)
      throws Exception {
    when(actionExecutionContext.getContext(eq(SpawnCache.class))).thenReturn(SpawnCache.NO_CACHE);
    when(actionExecutionContext.getExecRoot()).thenReturn(execRoot);
    when(actionExecutionContext.getContext(eq(SpawnLogContext.class)))
        .thenReturn(
            new SpawnLogContext(
                execRoot, messageOutput, executionOptions, remoteOptions, SyscallCache.NO_CACHE));
    when(spawnRunner.exec(any(Spawn.class), any(SpawnExecutionContext.class)))
        .thenReturn(
            new SpawnResult.Builder()
                .setStatus(Status.NON_ZERO_EXIT)
                .setExitCode(23)
                .setFailureDetail(NON_ZERO_EXIT_DETAILS)
                .setRunnerName("runner")
                .build());
    when(actionExecutionContext.getInputMetadataProvider())
        .thenReturn(mock(InputMetadataProvider.class));
  }

  /** Returns a SpawnExec pre-populated with values for a default spawn */
  private static SpawnExec.Builder defaultSpawnExecBuilder(String cmd) {
    return SpawnExec.newBuilder()
        .addCommandArgs(cmd)
        .setRemotable(true)
        .setCacheable(true)
        .setProgressMessage("progress message")
        .setMnemonic("Mnemonic")
        .setRunner("runner")
        .setStatus("NON_ZERO_EXIT")
        .setExitCode(23)
        .setRemoteCacheable(true)
        .setWalltime(Duration.getDefaultInstance())
        .setTargetLabel("//dummy:label");
  }
}
