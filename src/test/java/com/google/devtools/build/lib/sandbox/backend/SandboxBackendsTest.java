// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.sandbox.backend;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.SpawnStrategyRegistry;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import com.google.devtools.common.options.Options;
import java.io.IOException;
import java.nio.file.Files;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the registration of backends as spawn strategies in {@link SandboxBackends}. */
@RunWith(JUnit4.class)
public final class SandboxBackendsTest {
  private final DigestUtil digestUtil =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
  private java.nio.file.Path dir;
  private Path outputBase;
  private CommandEnvironment env;
  private FakeBackend backend;

  @Before
  public void setUp() throws Exception {
    dir = Files.createTempDirectory(java.nio.file.Path.of("/tmp"), "sbk-reg-");
    // All backends that are launched are linked to this one.
    backend = FakeBackend.onUnixSocket(dir.resolve("s").toString());
    FileSystem fs = FileSystems.getNativeFileSystem(DigestHashFunction.SHA256);
    outputBase = fs.getPath(dir.toString()).getRelative("output_base");
    outputBase.createDirectoryAndParents();
    Path execRoot = outputBase.getRelative("execroot/_main");
    execRoot.createDirectoryAndParents();
    env = mock(CommandEnvironment.class);
    when(env.getSyscallCache()).thenReturn(SyscallCache.NO_CACHE);
    when(env.getExecRoot()).thenReturn(execRoot);
    when(env.getOutputBase()).thenReturn(outputBase);
    when(env.getClientEnv()).thenReturn(ImmutableMap.of());
    when(env.getWorkspaceName()).thenReturn("_main");
    when(env.getBuildRequestId()).thenReturn("build");
    when(env.getCommandId()).thenReturn(UUID.randomUUID());
  }

  @After
  public void tearDown() throws Exception {
    SandboxBackends.stopAll();
    backend.close();
    fs().getPath(dir.toString()).deleteTree();
  }

  private static FileSystem fs() {
    return FileSystems.getNativeFileSystem(DigestHashFunction.SHA256);
  }

  private String script(String name) throws IOException {
    java.nio.file.Path script = dir.resolve(name);
    Files.writeString(
        script,
        "#!/bin/sh\n"
            + "for a in \"$@\"; do case $a in --socket=*) socket=${a#--socket=};; esac; done\n"
            + "ln -s \""
            + dir.resolve("s")
            + "\" \"$socket\"\n"
            + "exec sleep 600\n");
    script.toFile().setExecutable(true);
    return script.toString();
  }

  private Set<SpawnRunner> register(
      SpawnStrategyRegistry.Builder registry,
      List<Map.Entry<String, String>> backends,
      List<Map.Entry<String, String>> options)
      throws IOException {
    Set<SpawnRunner> runners = new HashSet<>();
    SandboxBackends.register(
        env, backends, options, Options.getDefaults(ExecutionOptions.class), registry, runners);
    return runners;
  }

  @Test
  public void register_makesEachBackendAStrategyAndARunnerWithAScratchRoot() throws Exception {
    SpawnStrategyRegistry.Builder registry = SpawnStrategyRegistry.builder();
    registry.setDefaultStrategies(ImmutableList.of("one"));

    Set<SpawnRunner> runners =
        register(
            registry,
            ImmutableList.of(Map.entry("one", script("one")), Map.entry("two", script("two"))),
            ImmutableList.of());

    assertThat(runners.stream().map(SpawnRunner::getName)).containsExactly("one", "two");
    Spawn spawn = new SpawnBuilder("x").build();
    var strategies =
        registry.build().getStrategies(spawn.getResourceOwner(), spawn.getMnemonic(), null);
    assertThat(strategies).hasSize(1);
    assertThat(strategies.get(0)).isInstanceOf(SandboxBackendStrategy.class);
    assertThat(outputBase.getRelative("sandbox-backend").isDirectory()).isTrue();
    // Nothing is launched until a spawn needs the backend.
    assertThat(runners.stream().allMatch(r -> r.canExec(spawn))).isTrue();
  }

  @Test
  public void register_backendWithoutBinary_declinesSpawns() throws Exception {
    Set<SpawnRunner> runners =
        register(
            SpawnStrategyRegistry.builder(),
            ImmutableList.of(Map.entry("gone", "/nonexistent/backend")),
            ImmutableList.of());

    assertThat(runners.iterator().next().canExec(new SpawnBuilder("x").build())).isFalse();
  }

  @Test
  public void register_nothingConfigured_registersNothingAndStopsRunningBackends()
      throws Exception {
    BackendConfig config = new BackendConfig(script("one"), ImmutableList.of());
    SandboxBackendSession session =
        SandboxBackends.getOrStart("one", config, ImmutableMap.of(), digestUtil);
    assertThat(session.isAlive()).isTrue();

    Set<SpawnRunner> runners =
        register(SpawnStrategyRegistry.builder(), ImmutableList.of(), ImmutableList.of());

    assertThat(runners).isEmpty();
    assertThat(session.isAlive()).isFalse();
  }

  @Test
  public void register_changedOptionsStopTheBackendButSameConfigurationKeepsIt() throws Exception {
    String binary = script("one");
    SandboxBackendSession session =
        SandboxBackends.getOrStart(
            "one",
            BackendConfig.fromFlags(
                    ImmutableList.of(Map.entry("one", binary)),
                    ImmutableList.of(Map.entry("one", "--a")))
                .get("one"),
            ImmutableMap.of(),
            digestUtil);

    register(
        SpawnStrategyRegistry.builder(),
        ImmutableList.of(Map.entry("one", binary)),
        ImmutableList.of(Map.entry("one", "--a")));
    assertThat(session.isAlive()).isTrue();

    register(
        SpawnStrategyRegistry.builder(),
        ImmutableList.of(Map.entry("one", binary)),
        ImmutableList.of(Map.entry("one", "--b")));
    assertThat(session.isAlive()).isFalse();
  }

  @Test
  public void register_invalidConfiguration_isAnError() {
    IOException e =
        assertThrows(
            IOException.class,
            () ->
                register(
                    SpawnStrategyRegistry.builder(),
                    ImmutableList.of(Map.entry("one", "/bin/true")),
                    ImmutableList.of(Map.entry("other", "--a"))));

    assertThat(e).hasMessageThat().contains("unknown sandbox backend 'other'");
  }
}
