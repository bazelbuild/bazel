// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.testutil.TestUtils.tmpDirFile;
import static java.nio.charset.StandardCharsets.UTF_8;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionUploadFinishedEvent;
import com.google.devtools.build.lib.authandtls.credentialhelper.CredentialModule;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.IntegrationTestUtils;
import com.google.devtools.build.lib.remote.util.IntegrationTestUtils.WorkerInstance;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlockWaitingModule;
import com.google.devtools.build.lib.runtime.BuildSummaryStatsModule;
import com.google.devtools.build.lib.standalone.StandaloneModule;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SyscallCache;
import java.io.IOException;
import java.time.Instant;
import java.util.ArrayDeque;
import java.util.Collection;
import java.util.HashSet;
import org.junit.After;
import org.junit.Before;
import org.junit.ClassRule;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class DiskCacheIntegrationTest extends BuildIntegrationTestCase {
  // Collect digests of AC entries uploaded to the disk cache during the build.
  // Filter out actions other than genrules (e.g. WorkspaceStatusAction).
  private final HashSet<Digest> actionDigests = new HashSet<>();
  private final ExtendedEventHandler actionDigestCollector =
      new ExtendedEventHandler() {
        @Override
        public void post(Postable obj) {
          if (obj instanceof ActionUploadFinishedEvent event) {
            if (!event.action().getMnemonic().equals("Genrule") || event.store() != Store.AC) {
              return;
            }
            actionDigests.add(event.digest());
          }
        }

        @Override
        public void handle(Event event) {}
      };

  private DigestUtil digestUtil;

  @ClassRule @Rule public static final WorkerInstance worker = IntegrationTestUtils.createWorker();

  private void enableRemoteExec(String... additionalOptions) {
    addOptions("--remote_executor=grpc://localhost:" + worker.getPort());
    addOptions(additionalOptions);
  }

  private void enableRemoteCache(String... additionalOptions) {
    addOptions("--remote_cache=grpc://localhost:" + worker.getPort());
    addOptions(additionalOptions);
  }

  private static PathFragment getDiskCacheDir() {
    PathFragment testTmpDir = PathFragment.create(tmpDirFile().getAbsolutePath());
    return testTmpDir.getRelative("disk_cache");
  }

  @Override
  protected void setupOptions() throws Exception {
    super.setupOptions();

    addOptions("--disk_cache=" + getDiskCacheDir());
  }

  @Before
  public void setUp() throws Exception {
    digestUtil = new DigestUtil(SyscallCache.NO_CACHE, getFileSystem().getDigestFunction());
    events.addHandler(actionDigestCollector);
  }

  @After
  public void tearDown() throws IOException {
    getWorkspace().getFileSystem().getPath(getDiskCacheDir()).deleteTree();
  }

  @Override
  protected ImmutableList<BlazeModule> getSpawnModules() {
    return ImmutableList.<BlazeModule>builder()
        .addAll(super.getSpawnModules())
        .add(new StandaloneModule())
        .build();
  }

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(new CredentialModule())
        .addBlazeModule(new RemoteModule())
        .addBlazeModule(new BuildSummaryStatsModule())
        .addBlazeModule(new BlockWaitingModule());
  }

  @Test
  public void hitDiskCache() throws Exception {
    // Arrange: Prepare the workspace and populate disk cache.
    setupWorkspace();
    buildTarget("//:foobar");
    assertThat(actionDigests).hasSize(2);
    assertRecentlyModified(actionDigests, getBlobDigests("foo", "foobar", "out", "err"));

    // Act: Reset mtime on cache entries and do a clean build.
    resetRecentlyModified();
    cleanAndRestartServer();
    buildTarget("//:foobar");

    // Assert: Should download action results from cache and refresh mtime on cache entries.
    events.assertContainsInfo("2 disk cache hit");
    assertRecentlyModified(actionDigests, getBlobDigests("foo", "foobar", "out", "err"));
  }

  private void doBlobsReferencedInAcAreMissingFromCasIgnoresAc(String... additionalOptions)
      throws Exception {
    // Arrange: Prepare the workspace and populate disk cache.
    setupWorkspace();
    addOptions(additionalOptions);
    buildTarget("//:foobar");

    // Act: Delete blobs in CAS from disk cache and do a clean build.
    getWorkspace().getFileSystem().getPath(getDiskCacheDir().getRelative("cas")).deleteTree();
    cleanAndRestartServer();
    addOptions(additionalOptions);
    buildTarget("//:foobar");

    // Assert: Should ignore the stale AC and rerun the generating action.
    events.assertDoesNotContainEvent("disk cache hit");
  }

  @Test
  public void blobsReferencedInAcAreMissingFromCas_ignoresAc() throws Exception {
    doBlobsReferencedInAcAreMissingFromCasIgnoresAc();
  }

  @Test
  public void bwob_blobsReferencedInAcAreMissingFromCas_ignoresAc() throws Exception {
    doBlobsReferencedInAcAreMissingFromCasIgnoresAc("--remote_download_minimal");
  }

  @Test
  public void bwobAndRemoteExec_blobsReferencedInAcAreMissingFromCas_ignoresAc() throws Exception {
    enableRemoteExec("--remote_download_minimal");
    doBlobsReferencedInAcAreMissingFromCasIgnoresAc();
  }

  @Test
  public void bwobAndRemoteCache_blobsReferencedInAcAreMissingFromCas_ignoresAc() throws Exception {
    enableRemoteCache("--remote_download_minimal");
    doBlobsReferencedInAcAreMissingFromCasIgnoresAc();
  }

  private void doRemoteExecWithDiskCache(String... additionalOptions) throws Exception {
    // Arrange: Prepare the workspace and populate disk cache.
    setupWorkspace();
    enableRemoteExec(additionalOptions);
    buildTarget("//:foobar");

    // Act: Do a clean build.
    cleanAndRestartServer();
    enableRemoteExec("--remote_download_minimal");
    buildTarget("//:foobar");
  }

  @Test
  public void remoteExecWithDiskCache_hitDiskCache() throws Exception {
    // Download all outputs to populate the disk cache.
    doRemoteExecWithDiskCache("--remote_download_all");

    // Assert: Should hit the disk cache.
    events.assertContainsInfo("2 disk cache hit");
  }

  @Test
  public void bwob_remoteExecWithDiskCache_hitRemoteCache() throws Exception {
    doRemoteExecWithDiskCache("--remote_download_minimal");

    // Assert: Should hit the remote cache because blobs referenced by the AC are missing from disk
    // cache due to BwoB.
    events.assertContainsInfo("2 remote cache hit");
  }

  @Test
  public void remoteExecWithDiskCache_inputsNotUploadedToDiskCache() throws Exception {
    // Arrange: Set up workspace with tree artifact, runfiles, and source file as inputs.
    write(
        "defs.bzl",
        """
        def _tree_impl(ctx):
            out = ctx.actions.declare_directory(ctx.attr.name + "_tree")
            ctx.actions.run_shell(
                mnemonic = "TreeGen",
                outputs = [out],
                command = "mkdir -p {0}/subdir && echo -n tree_content > {0}/subdir/file.txt".format(
                    out.path
                ),
            )
            return DefaultInfo(files = depset([out]))

        tree = rule(implementation = _tree_impl)

        def _runfiles_lib_impl(ctx):
            out = ctx.actions.declare_file(ctx.attr.name + ".txt")
            ctx.actions.write(out, "runfiles_content")
            return DefaultInfo(
                files = depset([out]),
                runfiles = ctx.runfiles(files = [out]),
            )

        runfiles_lib = rule(implementation = _runfiles_lib_impl)

        def _consumer_impl(ctx):
            out = ctx.actions.declare_file(ctx.attr.name + ".out")
            tree_input = ctx.attr.tree[DefaultInfo].files.to_list()[0]
            runfiles_input = ctx.attr.runfiles_lib[DefaultInfo].files.to_list()[0]
            source_input = ctx.file.src
            ctx.actions.run_shell(
                mnemonic = "Consumer",
                inputs = depset(
                    [tree_input, runfiles_input, source_input],
                    transitive = [ctx.attr.runfiles_lib[DefaultInfo].default_runfiles.files],
                ),
                outputs = [out],
                command = "cat {0}/subdir/file.txt {1} {2} > {3}".format(
                    tree_input.path, runfiles_input.path, source_input.path, out.path
                ),
            )
            return DefaultInfo(files = depset([out]))

        consumer = rule(
            implementation = _consumer_impl,
            attrs = {
                "tree": attr.label(mandatory = True),
                "runfiles_lib": attr.label(mandatory = True),
                "src": attr.label(mandatory = True, allow_single_file = True),
            },
        )
        """);
    write("source_input.txt", "source_content");
    write(
        "BUILD",
        """
        load(":defs.bzl", "tree", "runfiles_lib", "consumer")
        tree(name = "my_tree")
        runfiles_lib(name = "my_runfiles")
        consumer(
            name = "my_consumer",
            tree = ":my_tree",
            runfiles_lib = ":my_runfiles",
            src = "source_input.txt",
        )
        """);

    enableRemoteExec();
    buildTarget("//:my_consumer");

    // Assert: The tree artifact content, runfiles content, and source content should be in the
    // remote cache but NOT in the disk cache (inputs should only be uploaded to remote, not disk
    // cache).
    Digest treeContentDigest = digestUtil.compute("tree_content".getBytes(UTF_8));
    Digest runfilesContentDigest = digestUtil.compute("runfiles_content".getBytes(UTF_8));
    Digest sourceContentDigest = digestUtil.compute("source_content\n".getBytes(UTF_8));

    // Verify inputs are in the remote cache (so the remote action could execute).
    assertThat(remoteCacheEntryExists(treeContentDigest)).isTrue();
    assertThat(remoteCacheEntryExists(runfilesContentDigest)).isTrue();
    assertThat(remoteCacheEntryExists(sourceContentDigest)).isTrue();

    // Verify inputs are NOT in the disk cache.
    assertThat(diskCacheEntryExists(Store.CAS, treeContentDigest)).isFalse();
    assertThat(diskCacheEntryExists(Store.CAS, runfilesContentDigest)).isFalse();
    assertThat(diskCacheEntryExists(Store.CAS, sourceContentDigest)).isFalse();
  }

  private void cleanAndRestartServer() throws Exception {
    getOutputBase().getRelative("action_cache").deleteTreesBelow();
    // Simulates a server restart
    createRuntimeWrapper();
  }

  private void setupWorkspace() throws IOException {
    write(
        "BUILD",
        "genrule(",
        "  name = 'foo',",
        "  srcs = ['foo.in'],",
        "  outs = ['foo.out'],",
        "  cmd = 'echo -n foo > $@',",
        ")",
        "genrule(",
        "  name = 'foobar',",
        "  srcs = [':foo.out', 'bar.in'],",
        "  outs = ['foobar.out'],",
        "  cmd = 'echo -n out && echo -n err 1>&2 && echo -n foobar > $@',",
        ")");
    write("foo.in", "foo");
    write("bar.in", "bar");
  }

  private ImmutableSet<Digest> getBlobDigests(String... blobs) {
    // Uploaded CAS entries include the Action and Command protos which we don't care about,
    // so we can't collect them in the same manner as AC entries.
    ImmutableSet.Builder<Digest> digests = ImmutableSet.builder();
    for (String blob : blobs) {
      digests.add(digestUtil.compute(blob.getBytes(UTF_8)));
    }
    return digests.build();
  }

  private void resetRecentlyModified() throws IOException {
    ArrayDeque<Path> dirs = new ArrayDeque<>();
    dirs.add(getWorkspace().getFileSystem().getPath(getDiskCacheDir()));
    while (!dirs.isEmpty()) {
      Path dir = dirs.remove();
      for (Path child : dir.getDirectoryEntries()) {
        child.setLastModifiedTime(0);
        if (child.isDirectory()) {
          dirs.add(child);
        }
      }
    }
  }

  private void assertRecentlyModified(Collection<Digest> acDigests, Collection<Digest> casDigests)
      throws IOException {
    for (Digest digest : acDigests) {
      assertRecentlyModified(Store.AC, digest);
    }
    for (Digest digest : casDigests) {
      assertRecentlyModified(Store.CAS, digest);
    }
  }

  private void assertRecentlyModified(Store store, Digest digest) throws IOException {
    Path path = getDiskCacheEntryPath(store, digest);
    assertWithMessage("disk cache entry %s/%s does not exist", store, digest.getHash())
        .that(path.exists())
        .isTrue();
    assertWithMessage("disk cache entry %s/%s is too old", store, digest.getHash())
        .that(path.getLastModifiedTime())
        .isGreaterThan(Instant.now().minusSeconds(60).toEpochMilli());
  }

  private Path getDiskCacheEntryPath(Store store, Digest digest) throws IOException {
    return getWorkspace()
        .getFileSystem()
        .getPath(
            getDiskCacheDir()
                .getRelative(store.toString())
                .getRelative(digest.getHash().substring(0, 2))
                .getRelative(digest.getHash()));
  }

  private boolean diskCacheEntryExists(Store store, Digest digest) throws IOException {
    return getDiskCacheEntryPath(store, digest).exists();
  }

  private boolean remoteCacheEntryExists(Digest digest) {
    return fileSystem
        .getPath(
            worker
                .getCasPath()
                .getRelative("cas")
                .getRelative(digest.getHash().substring(0, 2))
                .getRelative(digest.getHash()))
        .exists();
  }
}
