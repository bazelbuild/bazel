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
import static org.junit.Assume.assumeTrue;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.ExecuteRequest;
import build.bazel.remote.execution.v2.OutputFile;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.dynamic.DynamicExecutionModule;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.longrunning.Operation;
import io.grpc.stub.ServerCallStreamObserver;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.TreeSet;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Dynamic execution of real spawns with sandbox backends on the local side, the remote side, or
 * both. Each backend is a {@link FakeBackend} on a real unix domain socket that is launched through
 * the production path.
 *
 * <p>Two backends, one on each side, are the only way to control the timing of both branches: a
 * standalone branch locks the outputs as soon as its process starts, so it beats a backend that has
 * to be launched and has to download its outputs. The tests with the standalone strategy therefore
 * only cover what happens when the backend loses.
 */
@RunWith(JUnit4.class)
public final class SandboxBackendDynamicExecutionIntegrationTest extends BuildIntegrationTestCase {
  private static final long WAIT_SECONDS = 30;

  private final DigestUtil digestUtil =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);

  /** A backend on its own socket, registered as strategy {@link #name}. */
  private final class Backend implements AutoCloseable {
    final String name;
    final Path dir;
    final FakeBackend fake;
    final CountDownLatch executing = new CountDownLatch(1);
    final CountDownLatch cancelled = new CountDownLatch(1);
    private int staged;

    Backend(String name) throws IOException {
      this.name = name;
      this.dir = Files.createTempDirectory(Path.of("/tmp"), "sbk-" + name + "-");
      this.fake = FakeBackend.onUnixSocket(dir.resolve("s").toString());
      Path script = dir.resolve("backend.sh");
      // Only makes the socket appear, by linking it to the socket of the server in this JVM.
      Files.writeString(
          script,
          "#!/bin/sh\n"
              + "for a in \"$@\"; do case $a in --socket=*) socket=${a#--socket=};; esac; done\n"
              + "ln -s \""
              + dir.resolve("s")
              + "\" \"$socket\"\n"
              + "exec sleep 600\n");
      script.toFile().setExecutable(true);
      addOptions("--sandbox_backend=" + name + "=" + script);
    }

    /** Holds the {@code Execute} call open until it is cancelled. */
    void hold() {
      fake.setExecuteHandler(
          (request, responses) -> {
            ((ServerCallStreamObserver<Operation>) responses).setOnCancelHandler(cancelled::countDown);
            executing.countDown();
          });
    }

    /** Succeeds with {@code contents} in each declared output, once {@code gate} has opened. */
    void succeedAfter(CountDownLatch gate, String contents) {
      fake.setExecuteHandler(
          (request, responses) -> {
            ((ServerCallStreamObserver<Operation>) responses).setOnCancelHandler(cancelled::countDown);
            executing.countDown();
            await(gate);
            FakeBackend.respond(responses, produce(request, contents));
          });
    }

    /** Fails the action, once {@code gate} has opened. */
    void failAfter(CountDownLatch gate) {
      fake.setExecuteHandler(
          (request, responses) -> {
            executing.countDown();
            await(gate);
            FakeBackend.respond(responses, 13, "backend broke");
          });
    }

    private ActionResult produce(ExecuteRequest request, String contents) throws IOException {
      Action action = Action.parseFrom(fake.stored.get(request.getActionDigest().getHash()));
      Command command = Command.parseFrom(fake.stored.get(action.getCommandDigest().getHash()));
      ActionResult.Builder result = ActionResult.newBuilder();
      Digest digest = digestUtil.computeAsUtf8(contents);
      for (String output : command.getOutputPathsList()) {
        Path file = dir.resolve("staged-" + staged++);
        Files.writeString(file, contents);
        fake.serve(digest, FileUriBlobs.encode(file.toString()));
        result.addOutputFiles(OutputFile.newBuilder().setPath(output).setDigest(digest));
      }
      return result.build();
    }

    boolean downloaded() {
      return fake.rpcs.contains("google.bytestream.ByteStream/Read");
    }

    @Override
    public void close() throws IOException {
      fake.close();
      try (var files = Files.walk(dir)) {
        files.sorted(Comparator.reverseOrder()).map(Path::toFile).forEach(f -> f.delete());
      }
    }
  }

  private final List<Backend> backends = new ArrayList<>();

  @Override
  protected ImmutableList<BlazeModule> getSpawnModules() {
    return ImmutableList.<BlazeModule>builder()
        .addAll(super.getSpawnModules())
        .add(new DynamicExecutionModule())
        .build();
  }

  @Before
  public void setUpDynamicExecution() {
    // Backends can't be reached on Windows yet, and these tests use unix sockets and shell scripts.
    assumeTrue(SandboxBackendTransport.isSupported());
    addOptions("--internal_spawn_scheduler", "--spawn_strategy=dynamic");
  }

  @After
  public void stopBackends() throws Exception {
    SandboxBackends.stopAll();
    for (Backend backend : backends) {
      backend.close();
    }
  }

  private Backend backend(String name) throws IOException {
    Backend backend = new Backend(name);
    backends.add(backend);
    return backend;
  }

  /** Backend {@code local} runs the local branch and {@code remote} the remote branch. */
  private void twoBackends(Backend local, Backend remote) throws IOException {
    addOptions("--dynamic_local_strategy=" + local.name, "--dynamic_remote_strategy=" + remote.name);
    // Neither branch ever runs the command.
    genrule("exit 1");
  }

  private void genrule(String cmd) throws IOException {
    write("a/BUILD", "genrule(name = 'g', outs = ['out.txt'], cmd = \"" + cmd + "\")");
  }

  private static void await(CountDownLatch latch) throws InterruptedException {
    assertThat(latch.await(WAIT_SECONDS, TimeUnit.SECONDS)).isTrue();
  }

  private String output() throws Exception {
    return readContentAsLatin1String(getArtifacts("//a:g").get(0)).trim();
  }

  /** The files below the directory of the package in the output tree, as names. */
  private TreeSet<String> outputTreeOfPackage() throws IOException {
    var names = new TreeSet<String>();
    try (var files = Files.walk(Path.of(getCommandEnvironment().getExecRoot().getPathString(), "bazel-out"))) {
      files
          .filter(f -> f.toString().contains("/bin/a/") || f.toString().endsWith("/bin/a"))
          .forEach(f -> names.add(f.getFileName().toString()));
    }
    return names;
  }

  // (a) A backend wins.

  @Test
  public void remoteBackendWins_localBackendStillExecuting_itIsCancelled() throws Exception {
    Backend local = backend("l");
    Backend remote = backend("r");
    twoBackends(local, remote);
    local.hold();
    remote.succeedAfter(local.executing, "REMOTE");

    buildTarget("//a:g");

    assertThat(output()).isEqualTo("REMOTE");
    // GetCapabilities is called once per session. Neither GetActionResult, QueryWriteStatus nor
    // BatchUpdateBlobs is called.
    assertThat(remote.fake.capabilitiesCalls.get()).isEqualTo(1);
    assertThat(remote.fake.rpcs)
        .containsExactly(
            "build.bazel.remote.execution.v2.Capabilities/GetCapabilities",
            "build.bazel.remote.execution.v2.ContentAddressableStorage/FindMissingBlobs",
            "google.bytestream.ByteStream/Write",
            "google.bytestream.ByteStream/Read",
            "build.bazel.remote.execution.v2.Execution/Execute");
    // The server of the loser sees the cancellation, and nothing of it reaches the output tree.
    await(local.cancelled);
    assertThat(local.downloaded()).isFalse();
    assertThat(outputTreeOfPackage()).containsExactly("a", "out.txt");
  }

  @Test
  public void localBackendWins_remoteBackendStillExecuting_itIsCancelled() throws Exception {
    Backend local = backend("l");
    Backend remote = backend("r");
    twoBackends(local, remote);
    remote.hold();
    local.succeedAfter(remote.executing, "LOCAL");

    buildTarget("//a:g");

    assertThat(output()).isEqualTo("LOCAL");
    await(remote.cancelled);
    assertThat(remote.downloaded()).isFalse();
    assertThat(outputTreeOfPackage()).containsExactly("a", "out.txt");
  }

  // (b) The other branch wins while the backend is downloading its outputs, i.e., after it has
  // finished executing but before it has committed anything.

  private void loserIsDownloadingWhenOtherWins(boolean loserIsRemote) throws Exception {
    Backend local = backend("l");
    Backend remote = backend("r");
    twoBackends(local, remote);
    Backend loser = loserIsRemote ? remote : local;
    Backend winner = loserIsRemote ? local : remote;
    var loserIsDownloading = new CountDownLatch(1);
    var release = new CountDownLatch(1);
    loser.fake.setOnRead(
        () -> {
          loserIsDownloading.countDown();
          try {
            release.await(WAIT_SECONDS, TimeUnit.SECONDS);
          } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
          }
        });
    loser.succeedAfter(new CountDownLatch(0), "LOSER");
    winner.succeedAfter(loserIsDownloading, "WINNER");

    try {
      buildTarget("//a:g");
    } finally {
      release.countDown();
    }

    assertThat(output()).isEqualTo("WINNER");
    assertThat(loser.downloaded()).isTrue();
    assertThat(outputTreeOfPackage()).containsExactly("a", "out.txt");
  }

  @Test
  public void remoteBackendLoses_whileDownloadingOutputs_nothingIsCommitted() throws Exception {
    loserIsDownloadingWhenOtherWins(/* loserIsRemote= */ true);
  }

  @Test
  public void localBackendLoses_whileDownloadingOutputs_nothingIsCommitted() throws Exception {
    loserIsDownloadingWhenOtherWins(/* loserIsRemote= */ false);
  }

  // (c) A backend fails. Dynamic execution has no hook that lets the other branch take over (the
  // IgnoreFailureCheck of the strategy is not set in Bazel), so the failure is the result of the
  // action, as for a failing local or remote strategy, and the other branch is cancelled.

  private void backendFails(boolean failingIsRemote) throws Exception {
    Backend local = backend("l");
    Backend remote = backend("r");
    twoBackends(local, remote);
    Backend failing = failingIsRemote ? remote : local;
    Backend other = failingIsRemote ? local : remote;
    other.hold();
    failing.failAfter(other.executing);

    BuildFailedException e = assertThrows(BuildFailedException.class, () -> buildTarget("//a:g"));

    assertThat(e).hasMessageThat().contains("backend broke");
    await(other.cancelled);
    assertThat(other.downloaded()).isFalse();
    assertThat(outputTreeOfPackage()).doesNotContain("out.txt");
  }

  @Test
  public void remoteBackendFails_theActionFailsAndTheLocalBackendIsCancelled() throws Exception {
    backendFails(/* failingIsRemote= */ true);
  }

  @Test
  public void localBackendFails_theActionFailsAndTheRemoteBackendIsCancelled() throws Exception {
    backendFails(/* failingIsRemote= */ false);
  }

  // With the standalone strategy on the other side the backend always loses: the standalone branch
  // locks the outputs when its process starts.

  @Test
  public void standaloneWins_againstRemoteBackend_nothingOfTheBackendIsCommitted() throws Exception {
    Backend remote = backend("r");
    addOptions("--dynamic_local_strategy=standalone", "--dynamic_remote_strategy=r");
    genrule("echo STANDALONE > $@");
    remote.succeedAfter(new CountDownLatch(0), "BACKEND");

    buildTarget("//a:g");

    assertThat(output()).isEqualTo("STANDALONE");
    assertThat(outputTreeOfPackage()).containsExactly("a", "out.txt");
  }

  @Test
  public void standaloneWins_againstLocalBackend_nothingOfTheBackendIsCommitted() throws Exception {
    Backend local = backend("l");
    addOptions("--dynamic_local_strategy=l", "--dynamic_remote_strategy=standalone");
    genrule("echo STANDALONE > $@");
    local.succeedAfter(new CountDownLatch(0), "BACKEND");

    buildTarget("//a:g");

    assertThat(output()).isEqualTo("STANDALONE");
    assertThat(outputTreeOfPackage()).containsExactly("a", "out.txt");
  }

  @Test
  public void failingRemoteBackend_standaloneWinsFirst_resultIsUsed() throws Exception {
    Backend remote = backend("r");
    addOptions("--dynamic_local_strategy=standalone", "--dynamic_remote_strategy=r");
    genrule("echo STANDALONE > $@");
    remote.failAfter(new CountDownLatch(0));

    buildTarget("//a:g");

    assertThat(output()).isEqualTo("STANDALONE");
  }
}
