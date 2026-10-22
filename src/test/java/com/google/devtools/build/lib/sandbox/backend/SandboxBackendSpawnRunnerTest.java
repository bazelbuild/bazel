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

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.FileNode;
import build.bazel.remote.execution.v2.OutputDirectory;
import build.bazel.remote.execution.v2.OutputFile;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.exec.util.FakeActionInputFileCache;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.FakeSpawnExecutionContext;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import com.google.protobuf.ByteString;
import com.google.rpc.Code;
import java.io.IOException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SandboxBackendSpawnRunner} against a {@link FakeBackend}. */
@RunWith(JUnit4.class)
public final class SandboxBackendSpawnRunnerTest {
  private final DigestUtil digestUtil =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);

  private FileSystem fs;
  private Path root;
  private Path execRoot;
  private Path staged;
  private ArtifactRoot artifactRoot;
  private FakeBackend backend;
  private SandboxBackendSession session;
  private SandboxBackendSpawnRunner runner;
  private final FakeActionInputFileCache fileCache = new FakeActionInputFileCache();
  private Artifact input;

  @Before
  public void setUp() throws Exception {
    fs = FileSystems.getNativeFileSystem(DigestHashFunction.SHA256);
    root = fs.getPath(java.nio.file.Files.createTempDirectory("sbk-test").toString());
    execRoot = root.getRelative("execroot/_main");
    execRoot.createDirectoryAndParents();
    staged = root.getRelative("staged");
    staged.createDirectoryAndParents();
    artifactRoot = ArtifactRoot.asDerivedRoot(execRoot, ArtifactRoot.RootType.OUTPUT, "outputs");
    input = ActionsTestUtil.createArtifact(artifactRoot, "pkg/in.txt");
    input.getPath().getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(input.getPath(), "input");
    fileCache.put(input, FileArtifactValue.createForTesting(input));

    backend = new FakeBackend();
    session = SandboxBackendSession.forChannel("fake", backend::newChannel, digestUtil);
    runner =
        new SandboxBackendSpawnRunner(
            "fake",
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
    session.close();
    backend.close();
    root.deleteTree();
  }

  private Spawn spawn(String... outputs) {
    return new SpawnBuilder("cmd", "arg").withInputs(input).withOutputs(outputs).build();
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
        spawn, fileCache, execRoot, new FileOutErr(root.getRelative("stdout"), root.getRelative("stderr")));
  }

  private Digest stage(String name, String contents) throws IOException {
    Path file = staged.getRelative(name);
    FileSystemUtils.writeContentAsLatin1(file, contents);
    Digest digest = digestUtil.computeAsUtf8(contents);
    backend.serve(digest, FileUriBlobs.encode(file.getPathString()));
    return digest;
  }

  private static OutputFile outputFile(String path, Digest digest) {
    return OutputFile.newBuilder().setPath(path).setDigest(digest).build();
  }

  private String read(String execPath) throws IOException {
    return FileSystemUtils.readContent(execRoot.getRelative(execPath), java.nio.charset.StandardCharsets.UTF_8);
  }

  private Digest inputDigest() {
    return digestUtil.computeAsUtf8("input");
  }

  @Test
  public void success_uploadsFileUrisAndMaterializesOutputs() throws Exception {
    Digest out = stage("o1", "hello");
    backend.setExecuteHandler(
        (request, responses) ->
            FakeBackend.respond(
                responses,
                ActionResult.newBuilder()
                    .addOutputFiles(outputFile("outputs/out.txt", out))
                    .setStderrRaw(ByteString.copyFromUtf8("warn"))
                    .build()));
    Spawn spawn = spawn("outputs/out.txt");
    Context context = context(spawn);

    SpawnResult result = runner.exec(spawn, context);

    assertThat(result.status()).isEqualTo(SpawnResult.Status.SUCCESS);
    assertThat(read("outputs/out.txt")).isEqualTo("hello");
    assertThat(context.isLockOutputFilesCalled()).isTrue();
    // The staged file is left alone for the backend to clean up.
    assertThat(staged.getRelative("o1").exists()).isTrue();
    // The input file is uploaded as its URI, not its contents.
    assertThat(backend.stored.get(inputDigest().getHash()))
        .isEqualTo(FileUriBlobs.encode(input.getPath().getPathString()));
    var request = backend.executes.get(0);
    assertThat(request.getSkipCacheLookup()).isTrue();
    assertThat(request.getInstanceName()).isEmpty();
    // Actions and commands are real blobs.
    Action action = Action.parseFrom(backend.stored.get(request.getActionDigest().getHash()));
    Command command = Command.parseFrom(backend.stored.get(action.getCommandDigest().getHash()));
    assertThat(command.getArgumentsList()).containsExactly("cmd", "arg").inOrder();
    assertThat(command.getOutputPathsList()).containsExactly("outputs/out.txt");
    assertThat(context.getFileOutErr().errAsLatin1()).isEqualTo("warn");
  }

  @Test
  public void secondExecution_uploadsNothing() throws Exception {
    Digest out = stage("o1", "hello");
    backend.setExecuteHandler(
        (request, responses) ->
            FakeBackend.respond(
                responses,
                ActionResult.newBuilder().addOutputFiles(outputFile("outputs/out.txt", out)).build()));
    Spawn spawn = spawn("outputs/out.txt");
    runner.exec(spawn, context(spawn));
    int writes = backend.writes.size();
    int findMissingCalls = backend.findMissingCalls.get();

    runner.exec(spawn, context(spawn));

    assertThat(backend.writes).hasSize(writes);
    assertThat(backend.findMissingCalls.get()).isEqualTo(findMissingCalls);
    assertThat(backend.executes).hasSize(2);
  }

  @Test
  public void outputsWithSameDigest_shareStagedFileWithoutMovingIt() throws Exception {
    Digest out = stage("o1", "same");
    backend.setExecuteHandler(
        (request, responses) ->
            FakeBackend.respond(
                responses,
                ActionResult.newBuilder()
                    .addOutputFiles(outputFile("outputs/a.txt", out))
                    .addOutputFiles(outputFile("outputs/b.txt", out))
                    .build()));
    Spawn spawn = spawn("outputs/a.txt", "outputs/b.txt");

    runner.exec(spawn, context(spawn));

    assertThat(read("outputs/a.txt")).isEqualTo("same");
    assertThat(read("outputs/b.txt")).isEqualTo("same");
    assertThat(staged.getRelative("o1").exists()).isTrue();
  }

  @Test
  public void executableBitOfOutput_isHonoredWithoutChangingStagedFile() throws Exception {
    Digest out = stage("o1", "script");
    backend.setExecuteHandler(
        (request, responses) ->
            FakeBackend.respond(
                responses,
                ActionResult.newBuilder()
                    .addOutputFiles(outputFile("outputs/run.sh", out).toBuilder().setIsExecutable(true))
                    .build()));
    Spawn spawn = spawn("outputs/run.sh");

    runner.exec(spawn, context(spawn));

    assertThat(execRoot.getRelative("outputs/run.sh").isExecutable()).isTrue();
    assertThat(staged.getRelative("o1").isExecutable()).isFalse();
  }

  @Test
  public void outputThatIsARealBlobStartingWithFileScheme_isNotFollowed() throws Exception {
    ByteString contents = ByteString.copyFromUtf8("file:///etc/passwd");
    Digest digest = digestUtil.compute(contents);
    backend.serve(digest, contents);
    backend.setExecuteHandler(
        (request, responses) ->
            FakeBackend.respond(
                responses,
                ActionResult.newBuilder().addOutputFiles(outputFile("outputs/url.txt", digest)).build()));
    Spawn spawn = spawn("outputs/url.txt");

    runner.exec(spawn, context(spawn));

    assertThat(read("outputs/url.txt")).isEqualTo("file:///etc/passwd");
  }

  @Test
  public void outputUriLongerThanBlob_isFollowed() throws Exception {
    // The URI is much longer than the 10 byte blob it stands for.
    Path file = staged.getRelative("x");
    String contents = "0123456789";
    FileSystemUtils.writeContentAsLatin1(file, contents);
    ByteString uri = FileUriBlobs.encode(file.getPathString());
    Digest digest = digestUtil.computeAsUtf8(contents);
    backend.serve(digest, uri);
    backend.setExecuteHandler(
        (request, responses) ->
            FakeBackend.respond(
                responses,
                ActionResult.newBuilder().addOutputFiles(outputFile("outputs/x.txt", digest)).build()));
    Spawn spawn = spawn("outputs/x.txt");

    runner.exec(spawn, context(spawn));

    assertThat(read("outputs/x.txt")).isEqualTo(contents);
  }

  @Test
  public void treeOutput_isAssembledFromItsFiles() throws Exception {
    Digest file = stage("o1", "in tree");
    Tree tree =
        Tree.newBuilder()
            .setRoot(
                Directory.newBuilder()
                    .addFiles(FileNode.newBuilder().setName("f.txt").setDigest(file)))
            .build();
    Digest treeDigest = digestUtil.compute(tree.toByteArray());
    backend.serve(treeDigest, tree.toByteString());
    backend.setExecuteHandler(
        (request, responses) ->
            FakeBackend.respond(
                responses,
                ActionResult.newBuilder()
                    .addOutputDirectories(
                        OutputDirectory.newBuilder()
                            .setPath("outputs/tree")
                            .setTreeDigest(treeDigest))
                    .build()));
    Spawn spawn = spawn("outputs/tree");

    runner.exec(spawn, context(spawn));

    assertThat(read("outputs/tree/f.txt")).isEqualTo("in tree");
  }

  @Test
  public void missingBlobs_reuploadAndRetry() throws Exception {
    Digest out = stage("o1", "hello");
    var calls = new java.util.concurrent.atomic.AtomicInteger();
    backend.setExecuteHandler(
        (request, responses) -> {
          if (calls.getAndIncrement() == 0) {
            backend.evict(inputDigest().getHash());
            FakeBackend.respondMissing(responses);
          } else {
            FakeBackend.respond(
                responses,
                ActionResult.newBuilder().addOutputFiles(outputFile("outputs/out.txt", out)).build());
          }
        });
    Spawn spawn = spawn("outputs/out.txt");

    SpawnResult result = runner.exec(spawn, context(spawn));

    assertThat(result.status()).isEqualTo(SpawnResult.Status.SUCCESS);
    assertThat(backend.executes).hasSize(2);
    assertThat(backend.writes.stream().filter(inputDigest().getHash()::equals).count())
        .isEqualTo(2);
    assertThat(backend.stored).containsKey(inputDigest().getHash());
  }

  @Test
  public void nonZeroExit_isReportedAndOutputsAreStillCollected() throws Exception {
    Digest out = stage("o1", "partial");
    backend.setExecuteHandler(
        (request, responses) ->
            FakeBackend.respond(
                responses,
                ActionResult.newBuilder()
                    .setExitCode(3)
                    .addOutputFiles(outputFile("outputs/out.txt", out))
                    .build()));
    Spawn spawn = spawn("outputs/out.txt");

    SpawnResult result = runner.exec(spawn, context(spawn));

    assertThat(result.status()).isEqualTo(SpawnResult.Status.NON_ZERO_EXIT);
    assertThat(result.exitCode()).isEqualTo(3);
    assertThat(read("outputs/out.txt")).isEqualTo("partial");
  }

  @Test
  public void executeFailure_isAnExecutionFailure() throws Exception {
    backend.setExecuteHandler(
        (request, responses) -> FakeBackend.respond(responses, Code.INTERNAL_VALUE, "kaboom"));
    Spawn spawn = spawn("outputs/out.txt");

    SpawnResult result = runner.exec(spawn, context(spawn));

    assertThat(result.status()).isEqualTo(SpawnResult.Status.EXECUTION_FAILED);
    assertThat(result.getFailureMessage()).contains("kaboom");
    assertThat(execRoot.getRelative("outputs/out.txt").exists()).isFalse();
  }

  @Test
  public void deadlineExceeded_isTimeout() throws Exception {
    backend.setExecuteHandler(
        (request, responses) ->
            FakeBackend.respond(responses, Code.DEADLINE_EXCEEDED_VALUE, "too slow"));
    Spawn spawn = spawn("outputs/out.txt");

    assertThat(runner.exec(spawn, context(spawn)).status())
        .isEqualTo(SpawnResult.Status.TIMEOUT);
  }

  @Test
  public void interrupt_cancelsExecuteAndLeavesOutputsAlone() throws Exception {
    var started = new CountDownLatch(1);
    var cancelled = new CountDownLatch(1);
    backend.setExecuteHandler(
        (request, responses) -> {
          ((io.grpc.stub.ServerCallStreamObserver<com.google.longrunning.Operation>) responses)
              .setOnCancelHandler(cancelled::countDown);
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

    assertThat(cancelled.await(10, TimeUnit.SECONDS)).isTrue();
    assertThat(failure.get()).isInstanceOf(InterruptedException.class);
    assertThat(execRoot.getRelative("outputs/out.txt").exists()).isFalse();
  }

  @Test
  public void backendDyingDuringExecute_isAnExecutionFailure() throws Exception {
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

    backend.killServer();
    thread.join(10_000);

    assertThat(result.get().status()).isEqualTo(SpawnResult.Status.EXECUTION_FAILED);
  }

  @Test
  public void outputsAreNotWrittenWhenDownloadFails() throws Exception {
    // Nothing is served for the digest, so reading the output fails.
    Digest missing = digestUtil.computeAsUtf8("gone");
    backend.setExecuteHandler(
        (request, responses) ->
            FakeBackend.respond(
                responses,
                ActionResult.newBuilder().addOutputFiles(outputFile("outputs/out.txt", missing)).build()));
    Spawn spawn = spawn("outputs/out.txt");
    Context context = context(spawn);

    SpawnResult result = runner.exec(spawn, context);

    assertThat(result.status()).isEqualTo(SpawnResult.Status.EXECUTION_FAILED);
    assertThat(context.isLockOutputFilesCalled()).isFalse();
    assertThat(execRoot.getRelative("outputs/out.txt").exists()).isFalse();
  }

  @Test
  public void declinesSpawnsThatMustNotBeSandboxed() {
    var declining =
        new SandboxBackendSpawnRunner(
            "fake", () -> session, () -> true, execRoot, root, "_main", "b", "c");
    Spawn local = new SpawnBuilder("x").withExecutionInfo("local", "").build();

    assertThat(declining.canExec(local)).isFalse();
    assertThat(declining.canExec(spawn())).isTrue();
    assertThat(
            new SandboxBackendSpawnRunner(
                    "fake", () -> session, () -> false, execRoot, root, "_main", "b", "c")
                .canExec(spawn()))
        .isFalse();
  }
}
