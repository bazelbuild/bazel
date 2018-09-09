// Copyright 2018 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;
import static org.junit.Assert.fail;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.OutputFile;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.remote.AbstractRemoteActionCache.UploadManifest;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.DigestUtil.ActionKey;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link AbstractRemoteActionCache}. */
@RunWith(JUnit4.class)
public class AbstractRemoteActionCacheTests {

  private FileSystem fs;
  private Path execRoot;
  private final DigestUtil digestUtil = new DigestUtil(DigestHashFunction.SHA256);

  private static ListeningScheduledExecutorService retryService;

  @BeforeClass
  public static void beforeEverything() {
    retryService = MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
  }

  @Before
  public void setUp() throws Exception {
    fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    execRoot = fs.getPath("/execroot");
    execRoot.createDirectory();
  }

  @AfterClass
  public static void afterEverything() {
    retryService.shutdownNow();
  }

  @Test
  public void uploadSymlinkAsFile() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path link = fs.getPath("/execroot/link");
    Path target = fs.getPath("/execroot/target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    link.createSymbolicLink(target);

    UploadManifest um = new UploadManifest(digestUtil, result, execRoot, true);
    um.addFiles(ImmutableList.of(link));
    assertThat(um.getDigestToFile()).containsExactly(digestUtil.compute(target), link);

    assertThat(
            assertThrows(
                ExecException.class,
                () ->
                    new UploadManifest(digestUtil, result, execRoot, false)
                        .addFiles(ImmutableList.of(link))))
        .hasMessageThat()
        .contains("Only regular files and directories may be uploaded to a remote cache.");
  }

  @Test
  public void uploadSymlinkInDirectory() throws Exception {
    ActionResult.Builder result = ActionResult.newBuilder();
    Path dir = fs.getPath("/execroot/dir");
    dir.createDirectory();
    Path target = fs.getPath("/execroot/target");
    FileSystemUtils.writeContent(target, new byte[] {1, 2, 3, 4, 5});
    Path link = fs.getPath("/execroot/dir/link");
    link.createSymbolicLink(fs.getPath("/execroot/target"));

    UploadManifest um = new UploadManifest(digestUtil, result, fs.getPath("/execroot"), true);
    um.addFiles(ImmutableList.of(link));
    assertThat(um.getDigestToFile()).containsExactly(digestUtil.compute(target), link);

    assertThat(
            assertThrows(
                ExecException.class,
                () ->
                    new UploadManifest(digestUtil, result, execRoot, false)
                        .addFiles(ImmutableList.of(link))))
        .hasMessageThat()
        .contains("Only regular files and directories may be uploaded to a remote cache.");
  }

  @Test
  public void onErrorWaitForRemainingDownloadsToComplete() throws Exception {
    // If one or more downloads of output files / directories fail then the code should
    // wait for all downloads to have been completed before it tries to clean up partially
    // downloaded files.

    Path stdout = fs.getPath("/execroot/stdout");
    Path stderr = fs.getPath("/execroot/stderr");

    Map<Digest, ListenableFuture<byte[]>> downloadResults = new HashMap<>();
    Path file1 = fs.getPath("/execroot/file1");
    Digest digest1 = digestUtil.compute("file1".getBytes("UTF-8"));
    downloadResults.put(digest1, Futures.immediateFuture("file1".getBytes("UTF-8")));
    Path file2 = fs.getPath("/execroot/file2");
    Digest digest2 = digestUtil.compute("file2".getBytes("UTF-8"));
    downloadResults.put(digest2, Futures.immediateFailedFuture(new IOException("download failed")));
    Path file3 = fs.getPath("/execroot/file3");
    Digest digest3 = digestUtil.compute("file3".getBytes("UTF-8"));
    downloadResults.put(digest3, Futures.immediateFuture("file3".getBytes("UTF-8")));

    RemoteOptions options = new RemoteOptions();
    RemoteRetrier retrier = new RemoteRetrier(options, (e) -> false, retryService,
        Retrier.ALLOW_ALL_CALLS);
    List<ListenableFuture<?>> blockingDownloads = new ArrayList<>();
    AtomicInteger numSuccess = new AtomicInteger();
    AtomicInteger numFailures = new AtomicInteger();
    AbstractRemoteActionCache cache = new DefaultRemoteActionCache(options, digestUtil, retrier) {
      @Override
      public ListenableFuture<Void> downloadBlob(Digest digest, OutputStream out) {
        SettableFuture<Void> result = SettableFuture.create();
        Futures.addCallback(downloadResults.get(digest), new FutureCallback<byte[]>() {
          @Override
          public void onSuccess(byte[] bytes) {
            numSuccess.incrementAndGet();
            try {
              out.write(bytes);
              out.close();
              result.set(null);
            } catch (IOException e) {
              result.setException(e);
            }
          }

          @Override
          public void onFailure(Throwable throwable) {
            numFailures.incrementAndGet();
            result.setException(throwable);
          }
        }, MoreExecutors.directExecutor());
        return result;
      }

      @Override
      protected <T> T getFromFuture(ListenableFuture<T> f)
          throws IOException, InterruptedException {
        blockingDownloads.add(f);
        return Utils.getFromFuture(f);
      }
    };

    ActionResult result = ActionResult.newBuilder()
        .setExitCode(0)
        .addOutputFiles(OutputFile.newBuilder().setPath(file1.getPathString()).setDigest(digest1))
        .addOutputFiles(OutputFile.newBuilder().setPath(file2.getPathString()).setDigest(digest2))
        .addOutputFiles(OutputFile.newBuilder().setPath(file3.getPathString()).setDigest(digest3))
        .build();
    try {
      cache.download(result, execRoot, ImmutableMap.builder(), ImmutableMap.builder(), new FileOutErr(stdout, stderr));
      fail("Expected IOException");
    } catch (IOException e) {
      assertThat(numSuccess.get()).isEqualTo(2);
      assertThat(numFailures.get()).isEqualTo(1);
      assertThat(blockingDownloads).hasSize(3);
      assertThat(Throwables.getRootCause(e)).hasMessageThat().isEqualTo("download failed");
    }
  }

  private static class DefaultRemoteActionCache extends AbstractRemoteActionCache {

    public DefaultRemoteActionCache(RemoteOptions options,
        DigestUtil digestUtil, Retrier retrier) {
      super(options, digestUtil, retrier);
    }

    @Override
    public void ensureInputsPresent(
        TreeNodeRepository repository, Path execRoot, TreeNode root, Action action, Command command)
        throws IOException, InterruptedException {
      throw new UnsupportedOperationException();
    }

    @Nullable
    @Override
    ActionResult getCachedActionResult(ActionKey actionKey)
        throws IOException, InterruptedException {
      throw new UnsupportedOperationException();
    }

    @Override
    void upload(
        ActionKey actionKey,
        Action action,
        Command command,
        Path execRoot,
        Collection<Path> files,
        FileOutErr outErr,
        boolean uploadAction)
        throws ExecException, IOException, InterruptedException {
      throw new UnsupportedOperationException();
    }

    @Override
    protected ListenableFuture<Void> downloadBlob(Digest digest, OutputStream out) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void close() {
      throw new UnsupportedOperationException();
    }
  }
}
