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
package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assume.assumeTrue;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.exec.LocalJobserver;
import com.google.devtools.build.lib.exec.PosixJobserverBackend;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.util.OS;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class LocalJobserverTest {

  @Rule public final TemporaryFolder tmp = new TemporaryFolder();

  private final LocalJobserver jobserver = LocalJobserver.instance();

  @Before
  public void unixOnly() {
    assumeTrue(OS.getCurrent() != OS.WINDOWS);
  }

  @After
  public void stop() {
    jobserver.shutdown();
  }

  private static Spawn taggedSpawn() {
    return new SpawnBuilder("cmd")
        .withExecutionInfo(ExecutionRequirements.SUPPORTS_JOBSERVER, "")
        .build();
  }

  /** Configures against a fresh ResourceManager reporting no idle CPU, so the pool stays empty. */
  private File configureQuiet() throws IOException {
    File dir = tmp.newFolder("jobserver");
    jobserver.configure(new PosixJobserverBackend(dir.getPath()), new ResourceManager());
    return dir;
  }

  @Test
  public void injectsFifoMakeflagsForTaggedSpawn() throws Exception {
    File dir = configureQuiet();
    ImmutableMap<String, String> result =
        jobserver.maybeAddJobserver(ImmutableMap.of("PATH", "/usr/bin"), taggedSpawn());
    assertThat(result)
        .containsEntry(
            LocalJobserver.MAKEFLAGS,
            "--jobserver-auth=fifo:" + new File(dir, "fifo").getPath());
    assertThat(result).containsEntry("PATH", "/usr/bin");
  }

  @Test
  public void noInjectionForUntaggedSpawn() throws Exception {
    configureQuiet();
    ImmutableMap<String, String> env = ImmutableMap.of("PATH", "/usr/bin");
    assertThat(jobserver.maybeAddJobserver(env, new SpawnBuilder("cmd").build())).isEqualTo(env);
  }

  @Test
  public void doesNotOverrideMakeflagsTheSpawnAlreadySets() throws Exception {
    configureQuiet();
    ImmutableMap<String, String> env = ImmutableMap.of(LocalJobserver.MAKEFLAGS, "-j2");
    assertThat(jobserver.maybeAddJobserver(env, taggedSpawn())).isEqualTo(env);
  }

  @Test
  public void noInjectionWhenJobserverInactive() {
    // configure() was never called: the flag is off, so even a tagged spawn is untouched.
    ImmutableMap<String, String> env = ImmutableMap.of("PATH", "/usr/bin");
    assertThat(jobserver.maybeAddJobserver(env, taggedSpawn())).isEqualTo(env);
  }

  @Test
  public void injectionDoesNotLeakIntoSpawnDeclaredEnvironment() throws Exception {
    configureQuiet();
    Spawn spawn = taggedSpawn();
    ImmutableMap<String, String> unused = jobserver.maybeAddJobserver(ImmutableMap.of(), spawn);
    assertThat(spawn.getEnvironment()).doesNotContainKey(LocalJobserver.MAKEFLAGS);
  }

  @Test
  public void getFifoDirForEnv_returnsWritableDirWhenEnvReferencesJobserver() throws Exception {
    File dir = configureQuiet();
    ImmutableMap<String, String> env = jobserver.maybeAddJobserver(ImmutableMap.of(), taggedSpawn());
    assertThat(jobserver.getFifoDirForEnv(env)).isEqualTo(dir.getPath());
  }

  @Test
  public void getFifoDirForEnv_nullWhenEnvDoesNotReferenceJobserver() throws Exception {
    configureQuiet();
    assertThat(jobserver.getFifoDirForEnv(ImmutableMap.of("PATH", "/usr/bin"))).isNull();
  }

  @Test
  public void shutdownDeletesFifoAndDisablesInjection() throws Exception {
    File dir = configureQuiet();
    File fifo = new File(dir, "fifo");
    assertThat(fifo.exists()).isTrue();

    jobserver.shutdown();

    assertThat(fifo.exists()).isFalse();
    assertThat(jobserver.maybeAddJobserver(ImmutableMap.of(), taggedSpawn())).isEmpty();
  }

  @Test
  public void tokenPoolTracksIdleCpuAndHeldTokens() throws Exception {
    ResourceManager rm = new ResourceManager();
    rm.setAvailableResources(ResourceSet.create(/* memoryMb= */ 1000.0, /* cpu= */ 4.0, /* tests= */ 0));
    rm.resetResourceUsage();
    File dir = tmp.newFolder("jobserver");
    jobserver.configure(new PosixJobserverBackend(dir.getPath()), rm);

    try (RandomAccessFile client = new RandomAccessFile(new File(dir, "fifo"), "rw")) {
      FileInputStream in = new FileInputStream(client.getFD());
      FileOutputStream out = new FileOutputStream(client.getFD());

      assertThat(drainUpTo(in, 4)).isEqualTo(4);
      awaitOutstanding(4);
      Thread.sleep(2 * 100); // two poll cycles; the pool must not grow past the idle-CPU budget
      assertThat(jobserver.getOutstandingTokens()).isEqualTo(4);

      // Returning the tokens frees the budget again.
      out.write(new byte[] {'+', '+', '+', '+'});
      out.flush();
      awaitOutstanding(0);
    }
  }

  @Test
  public void deadManagerThreadStopsChargingHeldTokens() throws Exception {
    FaultyResourceManager rm = new FaultyResourceManager();
    File dir = tmp.newFolder("jobserver");
    jobserver.configure(new PosixJobserverBackend(dir.getPath()), rm);

    try (RandomAccessFile client = new RandomAccessFile(new File(dir, "fifo"), "rw")) {
      FileInputStream in = new FileInputStream(client.getFD());
      // A client takes the whole pool, so the manager is charging 4 held tokens as in-use CPU.
      assertThat(drainUpTo(in, 4)).isEqualTo(4);
      awaitOutstanding(4);

      // The manager thread now dies (its next ResourceManager poll throws). The stale held-token
      // charge must be cleared to 0, else ResourceManager#isCpuAvailable would keep subtracting
      // it from the CPU budget for the rest of the build. Without the fix this stays stuck at 4.
      rm.failNow();
      awaitOutstanding(0);
    }
  }

  @Test
  public void managerSurvivesFailureToReAdmitWaitingActions() throws Exception {
    NotifyThrowingResourceManager rm = new NotifyThrowingResourceManager();
    File dir = tmp.newFolder("jobserver");
    jobserver.configure(new PosixJobserverBackend(dir.getPath()), rm);

    try (RandomAccessFile client = new RandomAccessFile(new File(dir, "fifo"), "rw")) {
      FileInputStream in = new FileInputStream(client.getFD());
      FileOutputStream out = new FileOutputStream(client.getFD());

      assertThat(drainUpTo(in, 4)).isEqualTo(4);
      awaitOutstanding(4);

      // Returning tokens makes the manager call notifyResourcesFreed, which throws. That failure
      // belongs to the waiting action being re-admitted, not to the manager: the pool must keep
      // being tracked and resized afterwards. Without the fix the manager thread dies here and
      // outstandingTokens stays 0 forever.
      out.write(new byte[] {'+', '+', '+', '+'});
      out.flush();
      awaitOutstanding(0);
      assertThat(rm.notifyCalls).isGreaterThan(0);

      assertThat(drainUpTo(in, 4)).isEqualTo(4);
      awaitOutstanding(4); // liveness: only a running manager re-derives the held-token count
    }
  }

  /** Drains up to {@code max} tokens, waiting up to 10s for the manager to fill the pool. */
  private static int drainUpTo(FileInputStream in, int max) throws IOException, InterruptedException {
    byte[] buf = new byte[max];
    AtomicInteger got = new AtomicInteger();
    AtomicReference<IOException> error = new AtomicReference<>();
    Thread reader =
        new Thread(
            () -> {
              try {
                int off = 0;
                while (off < max) {
                  int n = in.read(buf, off, max - off);
                  if (n < 0) {
                    break;
                  }
                  off += n;
                }
                got.set(off);
              } catch (IOException e) {
                error.set(e);
              }
            },
            "fifo-drain");
    reader.setDaemon(true);
    reader.start();
    reader.join(10_000);
    if (reader.isAlive()) {
      reader.interrupt();
      throw new IOException("timeout draining jobserver fifo");
    }
    if (error.get() != null) {
      throw error.get();
    }
    return got.get();
  }

  private void awaitOutstanding(int expected) throws InterruptedException {
    long deadline = System.currentTimeMillis() + 5_000;
    while (jobserver.getOutstandingTokens() != expected && System.currentTimeMillis() < deadline) {
      Thread.sleep(20);
    }
    assertThat(jobserver.getOutstandingTokens()).isEqualTo(expected);
  }

  /** A ResourceManager whose waiting-request re-admission always throws. */
  private static final class NotifyThrowingResourceManager extends ResourceManager {
    volatile int notifyCalls = 0;

    @Override
    public synchronized void notifyResourcesFreed() throws IOException {
      notifyCalls++;
      throw new IOException("injected re-admission failure");
    }

    @Override
    public synchronized double getIdleCpuForJobserver() {
      return 4.0;
    }
  }

  /** A ResourceManager whose jobserver idle-CPU poll can be made to throw, killing the manager. */
  private static final class FaultyResourceManager extends ResourceManager {
    private volatile boolean fail = false;

    void failNow() {
      fail = true;
    }

    @Override
    public synchronized double getIdleCpuForJobserver() {
      if (fail) {
        throw new IllegalStateException("injected jobserver manager fault");
      }
      return 4.0;
    }
  }
}
