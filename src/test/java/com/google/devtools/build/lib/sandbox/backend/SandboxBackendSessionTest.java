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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.SyscallCache;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the process lifecycle of {@link SandboxBackendSession} and {@link SandboxBackends}. */
@RunWith(JUnit4.class)
public final class SandboxBackendSessionTest {
  private final DigestUtil digestUtil =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
  private Path dir;
  private FakeBackend backend;

  @Before
  public void setUp() throws Exception {
    // Unix socket paths are short, so stay out of the long test temporary directory.
    dir = Files.createTempDirectory(Path.of("/tmp"), "sbk-session-");
    backend = FakeBackend.onUnixSocket(dir.resolve("s").toString());
  }

  @After
  public void tearDown() {
    SandboxBackends.stopAll();
    backend.close();
  }

  /**
   * A backend that creates its socket, as the backend contract requires, by linking it to the
   * socket of the in-JVM backend, then runs {@code body}.
   */
  private String script(String name, String body) throws IOException {
    Path script = dir.resolve(name);
    Files.writeString(
        script,
        "#!/bin/sh\n"
            + "for a in \"$@\"; do case $a in --socket=*) socket=${a#--socket=};; esac; done\n"
            + "ln -s \""
            + dir.resolve("s")
            + "\" \"$socket\"\n"
            + body
            + "\n");
    script.toFile().setExecutable(true);
    return script.toString();
  }

  private BackendConfig config(String binary, String... options) {
    return new BackendConfig(binary, ImmutableList.copyOf(options));
  }

  @Test
  public void start_waitsForSocket_andClosingKillsTheProcess() throws Exception {
    String binary = script("ok", "exec sleep 600");

    SandboxBackendSession session =
        SandboxBackendSession.start("ok", config(binary), ImmutableMap.of(), digestUtil);

    assertThat(session.isAlive()).isTrue();
    session.close();
    assertThat(session.isAlive()).isFalse();
  }

  @Test
  public void start_backendExitsBeforeCreatingSocket_fails() throws Exception {
    Path script = dir.resolve("bad");
    Files.writeString(script, "#!/bin/sh\necho boom\nexit 3\n");
    script.toFile().setExecutable(true);

    IOException e =
        assertThrows(
            IOException.class,
            () ->
                SandboxBackendSession.start(
                    "bad", config(script.toString()), ImmutableMap.of(), digestUtil));

    assertThat(e).hasMessageThat().contains("exited with status 3");
  }

  @Test
  public void start_missingBinary_fails() {
    IOException e =
        assertThrows(
            IOException.class,
            () ->
                SandboxBackendSession.start(
                    "none", config("/nonexistent/backend"), ImmutableMap.of(), digestUtil));

    assertThat(e).hasMessageThat().contains("doesn't exist or isn't executable");
  }

  @Test
  public void backendDying_isNoticed() throws Exception {
    String binary = script("dies", "echo goodbye\nsleep 1\nexit 7");
    SandboxBackendSession session =
        SandboxBackendSession.start("dies", config(binary), ImmutableMap.of(), digestUtil);

    long deadline = System.nanoTime() + 20_000_000_000L;
    while (session.isAlive() && System.nanoTime() < deadline) {
      Thread.sleep(20);
    }

    assertThat(session.isAlive()).isFalse();
    assertThat(session.deathMessage()).contains("exited with status 7");
    session.close();
  }

  @Test
  public void optionsAndSocketArePassedOnTheCommandLine() throws Exception {
    Path argv = dir.resolve("argv");
    String binary = script("args", "echo \"$@\" > " + argv + "\nexec sleep 600");

    SandboxBackendSession session =
        SandboxBackendSession.start(
            "args", config(binary, "--a=1", "b"), ImmutableMap.of(), digestUtil);

    long deadline = System.nanoTime() + 20_000_000_000L;
    while (!Files.exists(argv) && System.nanoTime() < deadline) {
      Thread.sleep(20);
    }
    String line = Files.readString(argv).trim();
    assertThat(line).startsWith("serve --socket=");
    assertThat(line).endsWith("--a=1 b");
    session.close();
  }

  @Test
  public void getOrStart_reusesSessionUntilConfigChanges() throws Exception {
    String binary = script("reuse", "exec sleep 600");
    BackendConfig config = config(binary);

    SandboxBackendSession first =
        SandboxBackends.getOrStart("reuse", config, ImmutableMap.of(), digestUtil);
    SandboxBackendSession again =
        SandboxBackends.getOrStart("reuse", config, ImmutableMap.of(), digestUtil);
    assertThat(again).isSameInstanceAs(first);

    SandboxBackendSession changed =
        SandboxBackends.getOrStart("reuse", config(binary, "--new"), ImmutableMap.of(), digestUtil);

    assertThat(changed).isNotSameInstanceAs(first);
    assertThat(first.isAlive()).isFalse();
    assertThat(changed.isAlive()).isTrue();
  }

  @Test
  public void reconcile_stopsBackendsThatAreNoLongerConfigured() throws Exception {
    String binary = script("gone", "exec sleep 600");
    SandboxBackendSession session =
        SandboxBackends.getOrStart("gone", config(binary), ImmutableMap.of(), digestUtil);

    SandboxBackends.reconcile(ImmutableMap.of());

    assertThat(session.isAlive()).isFalse();
  }

  @Test
  public void getOrStart_restartsDeadBackend() throws Exception {
    String binary = script("flaky", "sleep 1");
    BackendConfig config = config(binary);
    SandboxBackendSession first =
        SandboxBackends.getOrStart("flaky", config, ImmutableMap.of(), digestUtil);
    long deadline = System.nanoTime() + 20_000_000_000L;
    while (first.isAlive() && System.nanoTime() < deadline) {
      Thread.sleep(20);
    }

    SandboxBackendSession second =
        SandboxBackends.getOrStart("flaky", config, ImmutableMap.of(), digestUtil);

    assertThat(second).isNotSameInstanceAs(first);
  }
}
