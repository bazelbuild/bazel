// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.disk;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;
import static org.junit.Assume.assumeNotNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.FileNode;
import build.bazel.remote.execution.v2.OutputDirectory;
import build.bazel.remote.execution.v2.OutputFile;
import build.bazel.remote.execution.v2.RequestMetadata;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.exec.SpawnCheckingCacheEvent;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.remote.Store;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.OutputDigestMismatchException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.CachedActionResult;
import com.google.devtools.build.lib.remote.disk.Sqlite.Connection;
import com.google.devtools.build.lib.remote.disk.Sqlite.Result;
import com.google.devtools.build.lib.remote.disk.Sqlite.Statement;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.bazel.BazelHashFunctions;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.protobuf.ByteString;
import com.google.protobuf.Message;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import javax.annotation.Nullable;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DiskCacheClient}. */
@RunWith(JUnit4.class)
public class DiskCacheClientTest {
  private static final DigestUtil DIGEST_UTIL =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
  private static final ExecutorService executorService =
      MoreExecutors.listeningDecorator(Executors.newFixedThreadPool(1));

  private Path tmpDir;
  private Path root;
  private DiskCacheClient client;
  private RemoteActionExecutionContext context;

  @Before
  public void setUp() throws Exception {
    tmpDir = TestUtils.createUniqueTmpDir(null);
    root = tmpDir.getChild("disk_cache");
    client =
        new DiskCacheClient(
            root, /* maxSizeBytes= */ 0, DIGEST_UTIL, executorService, /* verifyDownloads= */ true);
    context =
        RemoteActionExecutionContext.create(
            mock(Spawn.class),
            mock(SpawnExecutionContext.class),
            RequestMetadata.getDefaultInstance());
  }

  @After
  public void tearDown() throws Exception {
    client.close();
  }

  @Test
  public void downloadActionResult_reportsSpawnCheckingCacheEvent() throws Exception {
    var unused =
        getFromFuture(
            client.downloadActionResult(
                context,
                DIGEST_UTIL.asActionKey(DIGEST_UTIL.computeAsUtf8("key")),
                /* inlineOutErr= */ false));

    verify(context.getSpawnExecutionContext()).report(SpawnCheckingCacheEvent.create("disk-cache"));
  }

  @Test
  public void findMissingDigests_returnsAllDigests() throws Exception {
    ImmutableList<Digest> digests =
        ImmutableList.of(getDigest("foo"), getDigest("bar"), getDigest("baz"));
    assertThat(getFromFuture(client.findMissingDigests(context, digests)))
        .containsExactlyElementsIn(digests);
  }

  @Test
  public void toPath_forCas_forOldStyleHashFunction() throws Exception {
    Digest digest = Digest.newBuilder().setHash("0123456789abcdef").setSizeBytes(42).build();

    Path path = client.toPath(digest, Store.CAS);

    assertThat(path).isEqualTo(root.getRelative("cas/01/0123456789abcdef"));
  }

  @Test
  public void toPath_forAc_forOldStyleHashFunction() throws Exception {
    Digest digest = Digest.newBuilder().setHash("0123456789abcdef").setSizeBytes(42).build();

    Path path = client.toPath(digest, Store.AC);

    assertThat(path).isEqualTo(root.getRelative("ac/01/0123456789abcdef"));
  }

  @Test
  public void toPath_forCas_forNewStyleHashFunction() throws Exception {
    assumeNotNull(BazelHashFunctions.BLAKE3); // BLAKE3 not available in Blaze.

    DiskCacheClient client =
        new DiskCacheClient(
            root,
            /* maxSizeBytes= */ 0,
            new DigestUtil(SyscallCache.NO_CACHE, BazelHashFunctions.BLAKE3),
            executorService,
            /* verifyDownloads= */ true);
    Digest digest = Digest.newBuilder().setHash("0123456789abcdef").setSizeBytes(42).build();
    Path path = client.toPath(digest, Store.CAS);

    assertThat(path).isEqualTo(root.getRelative("blake3/cas/01/0123456789abcdef"));
  }

  @Test
  public void toPath_forAc_forNewStyleHashFunction() throws Exception {
    assumeNotNull(BazelHashFunctions.BLAKE3); // BLAKE3 not available in Blaze.

    DiskCacheClient client =
        new DiskCacheClient(
            root,
            /* maxSizeBytes= */ 0,
            new DigestUtil(SyscallCache.NO_CACHE, BazelHashFunctions.BLAKE3),
            executorService,
            /* verifyDownloads= */ true);
    Digest digest = Digest.newBuilder().setHash("0123456789abcdef").setSizeBytes(42).build();
    Path path = client.toPath(digest, Store.AC);

    assertThat(path).isEqualTo(root.getRelative("blake3/ac/01/0123456789abcdef"));
  }

  @Test
  public void uploadFile_whenMissing_populatesCas() throws Exception {
    assertThat(root.exists()).isTrue();
    Path file = tmpDir.getChild("file");
    FileSystemUtils.writeContent(file, UTF_8, "contents");
    Digest digest = getDigest("contents");

    var unused = getFromFuture(client.uploadFile(context, digest, file));

    assertThat(FileSystemUtils.readContent(getCasPath(digest), UTF_8)).isEqualTo("contents");
  }

  @Test
  public void uploadFile_whenPresent_updatesMtime() throws Exception {
    Path file = tmpDir.getChild("file");
    FileSystemUtils.writeContent(file, UTF_8, "contents");
    Digest digest = getDigest("contents");

    // The contents would match under normal operation. This serves to check that we don't
    // unnecessarily overwrite the file.
    Path path = populateCas(digest, "existing contents");

    var unused = getFromFuture(client.uploadFile(context, digest, file));

    assertThat(FileSystemUtils.readContent(path, UTF_8)).isEqualTo("existing contents");
    assertThat(path.getLastModifiedTime()).isNotEqualTo(0);
  }

  @Test
  public void uploadBlob_whenMissing_populatesCas() throws Exception {
    ByteString blob = ByteString.copyFromUtf8("contents");
    Digest digest = getDigest("contents");

    var unused = getFromFuture(client.uploadBlob(context, digest, blob));

    assertThat(FileSystemUtils.readContent(getCasPath(digest), UTF_8)).isEqualTo("contents");
  }

  @Test
  public void uploadBlob_whenPresent_updatesMtime() throws Exception {
    ByteString blob = ByteString.copyFromUtf8("contents");
    Digest digest = getDigest("contents");

    // The contents would match under normal operation. This serves to check that we don't
    // unnecessarily overwrite the file.
    Path path = populateCas(digest, "existing contents");

    var unused = getFromFuture(client.uploadBlob(context, digest, blob));

    assertThat(FileSystemUtils.readContent(path, UTF_8)).isEqualTo("existing contents");
    assertThat(path.getLastModifiedTime()).isNotEqualTo(0);
  }

  @Test
  public void uploadActionResult_whenMissing_populatesAc() throws Exception {
    ActionKey actionKey = new ActionKey(getDigest("key"));
    ActionResult actionResult = ActionResult.newBuilder().setExitCode(42).build();
    Path path = getAcPath(actionKey);

    var unused = getFromFuture(client.uploadActionResult(context, actionKey, actionResult));

    assertThat(FileSystemUtils.readContent(path)).isEqualTo(actionResult.toByteArray());
  }

  @Test
  public void uploadActionResult_whenPresent_updatesMtime() throws Exception {
    ActionKey actionKey = new ActionKey(getDigest("key"));
    ActionResult actionResult = ActionResult.newBuilder().setExitCode(42).build();

    Path path = populateAc(actionKey, actionResult);

    // The contents would match under normal operation. This serves to check that we don't
    // unnecessarily overwrite the file.
    var unused =
        getFromFuture(
            client.uploadActionResult(context, actionKey, ActionResult.getDefaultInstance()));

    assertThat(FileSystemUtils.readContent(path)).isEqualTo(actionResult.toByteArray());
    assertThat(path.getLastModifiedTime()).isNotEqualTo(0);
  }

  @Test
  public void downloadBlob_whenPresent_returnsContents() throws Exception {
    Digest digest = getDigest("contents");
    populateCas(digest, "contents");
    Path out = tmpDir.getChild("out");

    var unused = getFromFuture(client.downloadBlob(context, digest, null, out.getOutputStream()));

    assertThat(FileSystemUtils.readContent(out, UTF_8)).isEqualTo("contents");
  }

  @Test
  public void downloadBlob_whenMissing_throwsCacheNotFoundException() throws Exception {
    Path out = tmpDir.getChild("out");

    assertThrows(
        CacheNotFoundException.class,
        () ->
            getFromFuture(
                client.downloadBlob(context, getDigest("contents"), null, out.getOutputStream())));
  }

  @Test
  public void downloadBlob_whenCorrupted_throwsOutputDigestMismatchException() throws Exception {
    Digest digest = getDigest("contents");
    populateCas(digest, "corrupted contents");
    Path out = tmpDir.getChild("out");

    assertThrows(
        OutputDigestMismatchException.class,
        () -> getFromFuture(client.downloadBlob(context, digest, null, out.getOutputStream())));
  }

  @Test
  public void downloadActionResult_whenPresent_returnsCachedActionResult() throws Exception {
    ActionKey actionKey = new ActionKey(getDigest("key"));
    ActionResult actionResult = ActionResult.newBuilder().setExitCode(42).build();

    Path path = populateAc(actionKey, actionResult);

    CachedActionResult result =
        getFromFuture(client.downloadActionResult(context, actionKey, /* inlineOutErr= */ false));

    assertThat(result).isEqualTo(CachedActionResult.disk(actionResult));
    assertThat(path.getLastModifiedTime()).isNotEqualTo(0);
  }

  @Test
  public void downloadActionResult_whenMissing_returnsNull() throws Exception {
    ActionKey actionKey = new ActionKey(getDigest("key"));

    CachedActionResult result =
        getFromFuture(client.downloadActionResult(context, actionKey, /* inlineOutErr= */ false));

    assertThat(result).isNull();
  }

  @Test
  public void downloadActionResult_withReferencedBlobsPresent_updatesMtimeOnBlobs()
      throws Exception {
    Digest stdoutDigest = getDigest("stdout contents");
    Digest stderrDigest = getDigest("stderr contents");
    Digest fileDigest = getDigest("file contents");
    Digest treeFileDigest = getDigest("tree file contents");
    Tree tree = getTreeWithFile(treeFileDigest);
    Digest treeDigest = getDigest(tree);
    ActionKey actionKey = new ActionKey(getDigest("key"));
    ActionResult actionResult =
        ActionResult.newBuilder()
            .setStdoutDigest(stdoutDigest)
            .setStderrDigest(stderrDigest)
            .addOutputFiles(OutputFile.newBuilder().setDigest(fileDigest).build())
            .addOutputDirectories(OutputDirectory.newBuilder().setTreeDigest(getDigest(tree)))
            .build();

    Path acPath = populateAc(actionKey, actionResult);
    Path stdoutCasPath = populateCas(stdoutDigest, "stdout contents");
    Path stderrCasPath = populateCas(stderrDigest, "stderr contents");
    Path fileCasPath = populateCas(fileDigest, "file contents");
    Path treeCasPath = populateCas(treeDigest, tree);
    Path treeFileCasPath = populateCas(treeFileDigest, "tree file contents");

    CachedActionResult result =
        getFromFuture(client.downloadActionResult(context, actionKey, /* inlineOutErr= */ false));

    assertThat(result).isEqualTo(CachedActionResult.disk(actionResult));
    assertThat(acPath.getLastModifiedTime()).isNotEqualTo(0);
    assertThat(stdoutCasPath.getLastModifiedTime()).isNotEqualTo(0);
    assertThat(stderrCasPath.getLastModifiedTime()).isNotEqualTo(0);
    assertThat(fileCasPath.getLastModifiedTime()).isNotEqualTo(0);
    assertThat(treeCasPath.getLastModifiedTime()).isNotEqualTo(0);
    assertThat(treeFileCasPath.getLastModifiedTime()).isNotEqualTo(0);
  }

  @Test
  public void downloadActionResult_withReferencedFileMissing_returnsNull() throws Exception {
    Digest fileDigest = getDigest("contents");
    ActionKey actionKey = new ActionKey(getDigest("key"));
    ActionResult actionResult =
        ActionResult.newBuilder()
            .addOutputFiles(OutputFile.newBuilder().setDigest(fileDigest).build())
            .build();

    populateAc(actionKey, actionResult);

    CachedActionResult result =
        getFromFuture(client.downloadActionResult(context, actionKey, /* inlineOutErr= */ false));

    assertThat(result).isNull();
  }

  @Test
  public void downloadActionResult_withReferencedTreeMissing_returnsNull() throws Exception {
    Digest fileDigest = getDigest("contents");
    Tree tree = getTreeWithFile(fileDigest);
    Digest treeDigest = getDigest(tree);
    ActionKey actionKey = new ActionKey(getDigest("key"));
    ActionResult actionResult =
        ActionResult.newBuilder()
            .addOutputDirectories(OutputDirectory.newBuilder().setTreeDigest(treeDigest))
            .build();

    populateAc(actionKey, actionResult);
    populateCas(fileDigest, "contents");

    CachedActionResult result =
        getFromFuture(client.downloadActionResult(context, actionKey, /* inlineOutErr= */ false));

    assertThat(result).isNull();
  }

  @Test
  public void downloadActionResult_withReferencedTreeFileMissing_returnsNull() throws Exception {
    Digest fileDigest = getDigest("contents");
    Tree tree = getTreeWithFile(fileDigest);
    Digest treeDigest = getDigest(tree);
    ActionKey actionKey = new ActionKey(getDigest("key"));
    ActionResult actionResult =
        ActionResult.newBuilder()
            .addOutputDirectories(OutputDirectory.newBuilder().setTreeDigest(treeDigest))
            .build();

    populateAc(actionKey, actionResult);
    populateCas(treeDigest, tree);

    CachedActionResult result =
        getFromFuture(client.downloadActionResult(context, actionKey, /* inlineOutErr= */ false));

    assertThat(result).isNull();
  }

  @Test
  public void ctor_withGcDisabled_withDbEmptyAndDirNonEmpty_doesNotCreateDb() throws Exception {
    writeFile(root.getRelative("cas/01/012345678"), "x".repeat(10), 123);
    writeFile(root.getRelative("ac/98/9876543210"), "x".repeat(20), 456);

    DiskCacheClient client =
        new DiskCacheClient(
            root, /* maxSizeBytes= */ 0, DIGEST_UTIL, executorService, /* verifyDownloads= */ true);

    client.close();

    assertThat(dumpDb()).isNull();
  }

  @Test
  public void ctor_withGcEnabled_withDbEmptyAndDirEmpty_createsDb() throws Exception {
    DiskCacheClient client =
        new DiskCacheClient(
            root,
            /* maxSizeBytes= */ 1024,
            DIGEST_UTIL,
            executorService,
            /* verifyDownloads= */ true);

    client.close();

    assertThat(dumpDb()).isEmpty();
  }

  @Test
  public void ctor_withGcEnabled_withDbEmptyAndDirNonEmpty_createsDb() throws Exception {
    writeFile(root.getRelative("cas/01/012345678"), "x".repeat(10), 123);
    writeFile(root.getRelative("ac/98/9876543210"), "x".repeat(20), 456);
    writeFile(root.getRelative("tmp/foo"), "", 789);

    DiskCacheClient client =
        new DiskCacheClient(
            root,
            /* maxSizeBytes= */ 1024,
            DIGEST_UTIL,
            executorService,
            /* verifyDownloads= */ true);

    client.close();

    assertThat(dumpDb())
        .containsExactly(
            new DbEntry("cas/01/012345678", 10, 123), new DbEntry("ac/98/9876543210", 20, 456));
  }

  private Tree getTreeWithFile(Digest fileDigest) {
    return Tree.newBuilder()
        .addChildren(Directory.newBuilder().addFiles(FileNode.newBuilder().setDigest(fileDigest)))
        .build();
  }

  private Path getCasPath(Digest digest) {
    return client.toPath(digest, Store.CAS);
  }

  @CanIgnoreReturnValue
  private Path populateCas(Digest digest, String contents) throws IOException {
    return populateCas(digest, contents.getBytes(UTF_8));
  }

  @CanIgnoreReturnValue
  private Path populateCas(Digest digest, Message m) throws IOException {
    return populateCas(digest, m.toByteArray());
  }

  private Path populateCas(Digest digest, byte[] contents) throws IOException {
    Path path = getCasPath(digest);
    writeFile(path, contents);
    return path;
  }

  private Path getAcPath(ActionKey actionKey) {
    return client.toPath(actionKey.getDigest(), Store.AC);
  }

  @CanIgnoreReturnValue
  private Path populateAc(ActionKey actionKey, ActionResult actionResult) throws IOException {
    Path path = getAcPath(actionKey);
    writeFile(path, actionResult.toByteArray());
    return path;
  }

  private void writeFile(Path path, byte[] contents) throws IOException {
    writeFile(path, contents, 0);
  }

  private void writeFile(Path path, String contents, long mtime) throws IOException {
    writeFile(path, contents.getBytes(UTF_8), mtime);
  }

  private void writeFile(Path path, byte[] contents, long mtime) throws IOException {
    path.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContent(path, contents);
    path.setLastModifiedTime(mtime);
  }

  private Digest getDigest(String contents) {
    return DIGEST_UTIL.computeAsUtf8(contents);
  }

  private Digest getDigest(Message m) {
    return DIGEST_UTIL.compute(m.toByteArray());
  }

  private record DbEntry(String name, long size, long mtime) {}

  @Nullable
  private ImmutableSet<DbEntry> dumpDb() throws IOException {
    Path dbPath = root.getRelative("gc/db.sqlite");
    if (!dbPath.exists()) {
      return null;
    }
    ImmutableSet.Builder<DbEntry> builder = ImmutableSet.builder();
    try (Connection conn = Sqlite.newConnection(dbPath);
        Statement stmt = conn.newStatement("SELECT path, size, mtime FROM entries");
        Result r = stmt.executeQuery()) {
      while (r.next()) {
        builder.add(new DbEntry(r.getString(0), r.getLong(1), r.getLong(2)));
      }
    }
    return builder.build();
  }
}
