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
package com.google.devtools.build.lib.windows;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.exec.LocalJobserver;
import com.google.devtools.build.lib.exec.WindowsJobserverBackend;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.util.OS;
import java.io.IOException;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Windows-only counterpart to {@code LocalJobserverTest}.
 */
@RunWith(JUnit4.class)
@TestSpec(supportedOs = OS.WINDOWS)
public final class WindowsJobserverTest {

  @Rule public final TemporaryFolder tmp = new TemporaryFolder();

  private long handle;
  private boolean hasHandle;

  @After
  public void tearDown() {
    if (hasHandle) {
      WindowsSemaphore.close(handle);
      hasHandle = false;
    }
    LocalJobserver.instance().shutdown();
  }

  private String uniqueName() {
    return "bazel-jobserver-test-" + System.nanoTime();
  }

  @Test
  public void releaseThenAcquireDrainsExactlyTheReleasedTokens() throws Exception {
    Long created = WindowsSemaphore.createSemaphore(uniqueName(), 4);
    assertThat(created).isNotNull();
    handle = created;
    hasHandle = true;

    assertThat(WindowsSemaphore.tryAcquire(handle)).isFalse();

    // Grow the pool by 4, then a tool can take exactly 4 and no more.
    assertThat(WindowsSemaphore.release(handle, 4)).isTrue();
    for (int i = 0; i < 4; i++) {
      assertThat(WindowsSemaphore.tryAcquire(handle)).isTrue();
    }
    assertThat(WindowsSemaphore.tryAcquire(handle)).isFalse();
  }

  @Test
  public void releaseBeyondMaxCountFails() {
    Long created = WindowsSemaphore.createSemaphore(uniqueName(), 2);
    assertThat(created).isNotNull();
    handle = created;
    hasHandle = true;

    assertThat(WindowsSemaphore.release(handle, 2)).isTrue();
    // The maximum count is fixed at creation (host CPU count in production); releasing past it must
    // fail rather than silently over-provision the pool.
    assertThat(WindowsSemaphore.release(handle, 1)).isFalse();
  }

  @Test
  public void aSecondHandleToTheSameNameSharesTheSameKernelSemaphore() throws Exception {
    String name = uniqueName();
    Long created = WindowsSemaphore.createSemaphore(name, 4);
    assertThat(created).isNotNull();
    handle = created;
    hasHandle = true;
    assertThat(WindowsSemaphore.release(handle, 2)).isTrue();

    // A tool opens the jobserver by name (CreateSemaphoreW returns a handle to the existing
    // object). It sees the two tokens the manager released.
    Long client = WindowsSemaphore.createSemaphore(name, 4);
    assertThat(client).isNotNull();
    long clientHandle = client;
    try {
      assertThat(WindowsSemaphore.tryAcquire(clientHandle)).isTrue();
      assertThat(WindowsSemaphore.tryAcquire(clientHandle)).isTrue();
      assertThat(WindowsSemaphore.tryAcquire(clientHandle)).isFalse();
    } finally {
      WindowsSemaphore.close(clientHandle);
    }
  }

  @Test
  public void tryAcquire_invalidHandleThrows() {
    assertThrows(IOException.class, () -> WindowsSemaphore.tryAcquire(0));
  }

  @Test
  public void backendRefreshesHeldCountOnSteadyTick() throws Exception {
    WindowsJobserverBackend backend =
        new WindowsJobserverBackend(tmp.newFolder("backend").getPath());
    String name = backend.start();
    Long client = WindowsSemaphore.createSemaphore(name, 1);
    assertThat(client).isNotNull();
    long clientHandle = client;
    try {
      assertThat(backend.tick(1)).isEqualTo(0);
      assertThat(WindowsSemaphore.tryAcquire(clientHandle)).isTrue();

      // The target did not shrink, but the backend must still observe the client's held token.
      assertThat(backend.tick(1)).isEqualTo(1);

      assertThat(WindowsSemaphore.release(clientHandle, 1)).isTrue();
      assertThat(backend.tick(1)).isEqualTo(0);
    } finally {
      WindowsSemaphore.close(clientHandle);
      backend.close();
    }
  }

  @Test
  public void injectsBareSemaphoreNameForTaggedSpawn() throws Exception {
    // A ResourceManager reporting no idle CPU keeps the manager thread quiet, so this asserts the
    // synchronously-set auth string without racing the poll loop.
    LocalJobserver.instance()
        .configure(
            new WindowsJobserverBackend(tmp.newFolder("jobserver").getPath()),
            new ResourceManager());

    Spawn tagged =
        new SpawnBuilder("cmd")
            .withExecutionInfo(ExecutionRequirements.SUPPORTS_JOBSERVER, "")
            .build();
    ImmutableMap<String, String> env =
        LocalJobserver.instance().maybeAddJobserver(ImmutableMap.of("PATH", "C:/tools"), tagged);

    String makeflags = env.get(LocalJobserver.MAKEFLAGS);
    assertThat(makeflags).startsWith("--jobserver-auth=bazel-jobserver-");
    assertThat(makeflags).doesNotContain("fifo:");
    assertThat(LocalJobserver.instance().getFifoDirForEnv(env)).isNull();
  }

  @Test
  public void shutdownDisablesInjection() throws Exception {
    LocalJobserver.instance()
        .configure(
            new WindowsJobserverBackend(tmp.newFolder("jobserver").getPath()),
            new ResourceManager());
    LocalJobserver.instance().shutdown();

    Spawn tagged =
        new SpawnBuilder("cmd")
            .withExecutionInfo(ExecutionRequirements.SUPPORTS_JOBSERVER, "")
            .build();
    assertThat(LocalJobserver.instance().maybeAddJobserver(ImmutableMap.of(), tagged)).isEmpty();
  }
}
