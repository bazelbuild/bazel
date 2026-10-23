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
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;
import static org.junit.Assume.assumeTrue;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.OutputFile;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.exec.util.FakeActionInputFileCache;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.Blob;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.FakeSpawnExecutionContext;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import com.google.longrunning.Operation;
import com.google.protobuf.ByteString;
import io.grpc.stub.ServerCallStreamObserver;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests that talk to a {@link FakeBackend} over a real unix domain socket, through the production
 * code that launches the backend process, waits for its socket and connects to it.
 *
 * <p>The backend process is a shell script that only links the socket that it is told to create to
 * the socket of the in-JVM server.
 */
@RunWith(JUnit4.class)
public final class SandboxBackendSocketTest {
  private static final String CAPABILITIES =
      "build.bazel.remote.execution.v2.Capabilities/GetCapabilities";

  private final DigestUtil digestUtil =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);

  private java.nio.file.Path tmp;
  private FileSystem fs;
  private Path root;
  private Path execRoot;
  private Path staged;
  private FakeBackend backend;
  private SandboxBackendSession session;
  private SandboxBackendSpawnRunner runner;
  private final FakeActionInputFileCache fileCache = new FakeActionInputFileCache();
  private Artifact input;

  @Before
  public void setUp() throws Exception {
    // Backends can't be reached on Windows yet, and these tests use unix sockets and shell scripts.
    assumeTrue(SandboxBackendTransport.isSupported());
    // Unix socket paths are short, so stay out of the long test temporary directory.
    tmp = java.nio.file.Files.createTempDirectory(java.nio.file.Path.of("/tmp"), "sbk-sock-");
    fs = FileSystems.getNativeFileSystem(DigestHashFunction.SHA256);
    root = fs.getPath(tmp.toString());
    execRoot = root.getRelative("execroot/_main");
    execRoot.createDirectoryAndParents();
    staged = root.getRelative("staged");
    staged.createDirectoryAndParents();
    ArtifactRoot artifactRoot =
        ArtifactRoot.asDerivedRoot(execRoot, ArtifactRoot.RootType.OUTPUT, "outputs");
    input = ActionsTestUtil.createArtifact(artifactRoot, "pkg/in.txt");
    input.getPath().getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(input.getPath(), "input");
    fileCache.put(input, FileArtifactValue.createForTesting(input));

    backend = FakeBackend.onUnixSocket(tmp.resolve("s").toString());
    java.nio.file.Path script = tmp.resolve("backend.sh");
    java.nio.file.Files.writeString(
        script,
        "#!/bin/sh\n"
            + "for a in \"$@\"; do case $a in --socket=*) socket=${a#--socket=};; esac; done\n"
            + "[ \"$1\" = serve ] || exit 64\n"
            + "echo $$ > \""
            + tmp.resolve("pid")
            + "\"\n"
            + "ln -s \""
            + tmp.resolve("s")
            + "\" \"$socket\"\n"
            + "exec sleep 600\n");
    script.toFile().setExecutable(true);
    // The production path: launch, poll for the socket, connect through GoogleAuthUtils.
    session =
        SandboxBackendSession.start(
            "sock",
            new BackendConfig(script.toString(), ImmutableList.of()),
            ImmutableMap.of(),
            digestUtil);
    runner =
        new SandboxBackendSpawnRunner(
            "sock",
            () -> session,
            () -> true,
            execRoot,
            root.getRelative("scratch"),
            "_main",
            "build",
            "command");
  }

  @After
  public void tearDown() throws Exception {
    if (session != null) {
      session.close();
    }
    if (backend != null) {
      backend.close();
    }
    if (root != null) {
      root.deleteTree();
    }
  }

  private static final class Context extends FakeSpawnExecutionContext {
    Context(Spawn spawn, FakeActionInputFileCache cache, Path execRoot, FileOutErr outErr) {
      super(
          spawn,
          cache,
          execRoot,
          outErr,
          ImmutableClassToInstanceMap.of(),
          /* actionFileSystem= */ null);
    }

    @Override
    public ListenableFuture<Void> prefetchInputs() {
      return immediateVoidFuture();
    }
  }

  private Context context(Spawn spawn) {
    return new Context(
        spawn,
        fileCache,
        execRoot,
        new FileOutErr(root.getRelative("stdout"), root.getRelative("stderr")));
  }

  private Spawn spawn(String... outputs) {
    return new SpawnBuilder("cmd", "arg").withInputs(input).withOutputs(outputs).build();
  }

  private RemoteActionExecutionContext remoteContext() {
    return RemoteActionExecutionContext.create(RequestMetadata.getDefaultInstance());
  }

  @Test
  public void findMissingBlobs_overUnixSocket() throws Exception {
    Digest present = digestUtil.computeAsUtf8("present");
    Digest absent = digestUtil.computeAsUtf8("absent");
    backend.stored.put(present.getHash(), ByteString.copyFromUtf8("present"));

    ImmutableSet<Digest> missing =
        session
            .cacheClient()
            .findMissingDigests(remoteContext(), ImmutableList.of(present, absent))
            .get(10, TimeUnit.SECONDS);

    assertThat(missing).containsExactly(absent);
    assertThat(backend.rpcs)
        .containsExactly(
            CAPABILITIES, "build.bazel.remote.execution.v2.ContentAddressableStorage/FindMissingBlobs");
  }

  @Test
  public void byteStreamWrite_fileUriAndRealBlob() throws Exception {
    ByteString uri = FileUriBlobs.encode(input.getPath().getPathString());
    Digest fileDigest = digestUtil.computeAsUtf8("input");
    Digest realDigest = digestUtil.computeAsUtf8("real bytes");

    session
        .cacheClient()
        .uploadBlob(
            remoteContext(),
            fileDigest,
            new Blob() {
              @Override
              public java.io.InputStream get() {
                return uri.newInput();
              }

              @Override
              public java.util.OptionalLong streamedSize() {
                return java.util.OptionalLong.of(uri.size());
              }
            },
            /* force= */ true)
        .get(10, TimeUnit.SECONDS);
    session
        .cacheClient()
        .uploadBlob(remoteContext(), realDigest, ByteString.copyFromUtf8("real bytes"), true)
        .get(10, TimeUnit.SECONDS);

    assertThat(backend.stored.get(fileDigest.getHash())).isEqualTo(uri);
    assertThat(backend.stored.get(realDigest.getHash())).isEqualTo(ByteString.copyFromUtf8("real bytes"));
    assertThat(backend.rpcs).containsExactly(CAPABILITIES, "google.bytestream.ByteStream/Write");
  }

  @Test
  public void byteStreamRead_fileUriAndRealBytes() throws Exception {
    Path file = staged.getRelative("f");
    FileSystemUtils.writeContentAsLatin1(file, "0123456789");
    Digest fileDigest = digestUtil.computeAsUtf8("0123456789");
    ByteString uri = FileUriBlobs.encode(file.getPathString());
    backend.serve(fileDigest, uri);
    Digest realDigest = digestUtil.computeAsUtf8("real");
    backend.serve(realDigest, ByteString.copyFromUtf8("real"));

    // Streamed reads don't check the size, which differs for a file URI.
    assertThat(download(fileDigest)).isEqualTo(uri);
    assertThat(download(realDigest)).isEqualTo(ByteString.copyFromUtf8("real"));
    assertThat(backend.rpcs).containsExactly(CAPABILITIES, "google.bytestream.ByteStream/Read");
  }

  private ByteString download(Digest digest) throws Exception {
    var out = new ByteArrayOutputStream();
    session
        .cacheClient()
        .downloadStreamedBytes(remoteContext(), digest, out)
        .get(10, TimeUnit.SECONDS);
    return ByteString.copyFrom(out.toByteArray());
  }

  private static OutputFile outputFile(String path, Digest digest) {
    return OutputFile.newBuilder().setPath(path).setDigest(digest).build();
  }

  @Test
  public void fullExecute_roundTrip_andObservedRpcs() throws Exception {
    Path out = staged.getRelative("o1");
    FileSystemUtils.writeContentAsLatin1(out, "hello");
    Digest outDigest = digestUtil.computeAsUtf8("hello");
    backend.serve(outDigest, FileUriBlobs.encode(out.getPathString()));
    backend.setExecuteHandler(
        (request, responses) ->
            FakeBackend.respond(
                responses,
                ActionResult.newBuilder()
                    .addOutputFiles(outputFile("outputs/out.txt", outDigest))
                    .setStdoutRaw(ByteString.copyFromUtf8("said"))
                    .build()));
    Spawn spawn = spawn("outputs/out.txt");
    Context context = context(spawn);

    SpawnResult result = runner.exec(spawn, context);

    assertThat(result.status()).isEqualTo(SpawnResult.Status.SUCCESS);
    assertThat(
            FileSystemUtils.readContent(
                execRoot.getRelative("outputs/out.txt"), StandardCharsets.UTF_8))
        .isEqualTo("hello");
    assertThat(context.getFileOutErr().outAsLatin1()).isEqualTo("said");
    assertThat(backend.stored.get(digestUtil.computeAsUtf8("input").getHash()))
        .isEqualTo(FileUriBlobs.encode(input.getPath().getPathString()));
    // Exactly what the contract in the design document promises, with GetCapabilities once per
    // session: no GetActionResult, QueryWriteStatus or BatchUpdateBlobs.
    assertThat(backend.capabilitiesCalls.get()).isEqualTo(1);
    assertThat(backend.rpcs)
        .containsExactly(
            CAPABILITIES,
            "build.bazel.remote.execution.v2.ContentAddressableStorage/FindMissingBlobs",
            "google.bytestream.ByteStream/Write",
            "google.bytestream.ByteStream/Read",
            "build.bazel.remote.execution.v2.Execution/Execute");
  }

  @Test
  public void interrupt_cancelsExecuteOnTheServer() throws Exception {
    var started = new CountDownLatch(1);
    var cancelled = new CountDownLatch(1);
    backend.setExecuteHandler(
        (request, responses) -> {
          ((ServerCallStreamObserver<Operation>) responses).setOnCancelHandler(cancelled::countDown);
          started.countDown();
        });
    Spawn spawn = spawn("outputs/out.txt");
    var failure = new AtomicReference<Throwable>();
    Thread thread =
        new Thread(
            () -> {
              try {
                runner.exec(spawn, context(spawn));
              } catch (Throwable t) {
                failure.set(t);
              }
            });
    thread.start();
    assertThat(started.await(10, TimeUnit.SECONDS)).isTrue();

    thread.interrupt();
    thread.join(10_000);

    // The server observes the client cancelling the call over the real socket.
    assertThat(cancelled.await(10, TimeUnit.SECONDS)).isTrue();
    assertThat(failure.get()).isInstanceOf(InterruptedException.class);
    assertThat(execRoot.getRelative("outputs/out.txt").exists()).isFalse();
  }

  @Test
  public void killingTheBackendProcess_failsTheCallInFlight() throws Exception {
    var started = new CountDownLatch(1);
    backend.setExecuteHandler((request, responses) -> started.countDown());
    Spawn spawn = spawn("outputs/out.txt");
    var result = new AtomicReference<SpawnResult>();
    Thread thread =
        new Thread(
            () -> {
              try {
                result.set(runner.exec(spawn, context(spawn)));
              } catch (Exception e) {
                throw new AssertionError(e);
              }
            });
    thread.start();
    assertThat(started.await(10, TimeUnit.SECONDS)).isTrue();

    // Only the process is killed, the server that answers on the socket stays up.
    long pid = Long.parseLong(java.nio.file.Files.readString(tmp.resolve("pid")).trim());
    assertThat(ProcessHandle.of(pid).orElseThrow().destroyForcibly()).isTrue();
    thread.join(20_000);

    assertThat(thread.isAlive()).isFalse();
    assertThat(result.get().status()).isEqualTo(SpawnResult.Status.EXECUTION_FAILED);
    assertThat(result.get().getFailureMessage()).contains("sandbox backend 'sock' exited with status");
    assertThat(session.isAlive()).isFalse();
  }
}
