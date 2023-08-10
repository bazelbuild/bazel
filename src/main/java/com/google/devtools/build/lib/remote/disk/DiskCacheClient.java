// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.remote.util.DigestUtil.isOldStyleDigestFunction;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.base.Ascii;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.ByteStreams;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.util.DigestOutputStream;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import com.google.protobuf.ExtensionRegistryLite;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.UUID;
import javax.annotation.Nullable;

/** A on-disk store for the remote action cache. */
public class DiskCacheClient implements RemoteCacheClient {

  private static final String AC_DIRECTORY = "ac";
  private static final String CAS_DIRECTORY = "cas";

  private final Path root;
  private final boolean verifyDownloads;
  private final DigestUtil digestUtil;

  /**
   * @param verifyDownloads whether verify the digest of downloaded content are the same as the
   *     digest used to index that file.
   */
  public DiskCacheClient(Path root, boolean verifyDownloads, DigestUtil digestUtil)
      throws IOException {
    this.verifyDownloads = verifyDownloads;
    this.digestUtil = digestUtil;

    if (isOldStyleDigestFunction(digestUtil.getDigestFunction())) {
      this.root = root;
    } else {
      this.root =
          root.getChild(
              Ascii.toLowerCase(digestUtil.getDigestFunction().getValueDescriptor().getName()));
    }

    this.root.createDirectoryAndParents();
  }

  /** Returns {@code true} if the provided {@code key} is stored in the CAS. */
  public boolean contains(Digest digest) {
    return toPath(digest.getHash(), /* actionResult= */ false).exists();
  }

  /** Returns {@link Path} into the CAS for the given {@link Digest}. */
  public Path getPath(Digest digest) {
    return toPath(digest.getHash(), /* actionResult= */ false);
  }

  public void captureFile(Path src, Digest digest, boolean isActionCache) throws IOException {
    Path target = toPath(digest.getHash(), isActionCache);
    target.getParentDirectory().createDirectoryAndParents();
    src.renameTo(target);
  }

  private ListenableFuture<Void> download(Digest digest, OutputStream out, boolean isActionCache) {
    Path p = toPath(digest.getHash(), isActionCache);
    if (!p.exists()) {
      return Futures.immediateFailedFuture(new CacheNotFoundException(digest));
    } else {
      try (InputStream in = p.getInputStream()) {
        ByteStreams.copy(in, out);
        return Futures.immediateFuture(null);
      } catch (IOException e) {
        return Futures.immediateFailedFuture(e);
      }
    }
  }

  @Override
  public ListenableFuture<Void> downloadBlob(
      RemoteActionExecutionContext context, Digest digest, OutputStream out) {
    @Nullable
    DigestOutputStream digestOut = verifyDownloads ? digestUtil.newDigestOutputStream(out) : null;
    return Futures.transformAsync(
        download(digest, digestOut != null ? digestOut : out, /* isActionCache= */ false),
        (v) -> {
          try {
            if (digestOut != null) {
              Utils.verifyBlobContents(digest, digestOut.digest());
            }
            out.flush();
            return Futures.immediateFuture(null);
          } catch (IOException e) {
            return Futures.immediateFailedFuture(e);
          }
        },
        MoreExecutors.directExecutor());
  }

  private void checkDigestExists(Digest digest) throws CacheNotFoundException {
    if (digest.getSizeBytes() == 0) {
      return;
    }

    if (!toPath(digest.getHash(), /* actionResult= */ false).exists()) {
      throw new CacheNotFoundException(digest);
    }
  }

  private void checkOutputDirectory(Directory dir) throws CacheNotFoundException {
    for (var file : dir.getFilesList()) {
      checkDigestExists(file.getDigest());
    }
  }

  private void checkActionResult(ActionResult actionResult) throws IOException {
    for (var outputFile : actionResult.getOutputFilesList()) {
      checkDigestExists(outputFile.getDigest());
    }

    for (var outputDirectory : actionResult.getOutputDirectoriesList()) {
      var treeDigest = outputDirectory.getTreeDigest();
      checkDigestExists(treeDigest);

      var treePath = toPath(treeDigest.getHash(), /* actionResult= */ false);
      var tree =
          Tree.parseFrom(treePath.getInputStream(), ExtensionRegistryLite.getEmptyRegistry());
      checkOutputDirectory(tree.getRoot());
      for (var dir : tree.getChildrenList()) {
        checkOutputDirectory(dir);
      }
    }
  }

  @Override
  public ListenableFuture<CachedActionResult> downloadActionResult(
      RemoteActionExecutionContext context, ActionKey actionKey, boolean inlineOutErr) {
    return Futures.transformAsync(
        Utils.downloadAsActionResult(
            actionKey, (digest, out) -> download(digest, out, /* isActionCache= */ true)),
        actionResult -> {
          if (actionResult == null) {
            return Futures.immediateFuture(null);
          }

          try {
            checkActionResult(actionResult);
          } catch (CacheNotFoundException e) {
            return Futures.immediateFuture(null);
          }

          return Futures.immediateFuture(CachedActionResult.disk(actionResult));
        },
        MoreExecutors.directExecutor());
  }

  @Override
  public ListenableFuture<Void> uploadActionResult(
      RemoteActionExecutionContext context, ActionKey actionKey, ActionResult actionResult) {
    try (InputStream data = actionResult.toByteString().newInput()) {
      saveFile(actionKey.getDigest().getHash(), data, /* actionResult= */ true);
      return Futures.immediateFuture(null);
    } catch (IOException e) {
      return Futures.immediateFailedFuture(e);
    }
  }

  @Override
  public void close() {}

  @Override
  public ListenableFuture<Void> uploadFile(
      RemoteActionExecutionContext context, Digest digest, Path file) {
    try (InputStream in = file.getInputStream()) {
      saveFile(digest.getHash(), in, /* actionResult= */ false);
    } catch (IOException e) {
      return Futures.immediateFailedFuture(e);
    }
    return Futures.immediateFuture(null);
  }

  @Override
  public ListenableFuture<Void> uploadBlob(
      RemoteActionExecutionContext context, Digest digest, ByteString data) {
    try (InputStream in = data.newInput()) {
      saveFile(digest.getHash(), in, /* actionResult= */ false);
    } catch (IOException e) {
      return Futures.immediateFailedFuture(e);
    }
    return Futures.immediateFuture(null);
  }

  @Override
  public ListenableFuture<ImmutableSet<Digest>> findMissingDigests(
      RemoteActionExecutionContext context, Iterable<Digest> digests) {
    // Both upload and download check if the file exists before doing I/O. So we don't
    // have to do it here.
    return Futures.immediateFuture(ImmutableSet.copyOf(digests));
  }

  protected Path toPathNoSplit(String key) {
    return root.getChild(key);
  }

  protected Path toPath(String key, boolean actionResult) {
    String cacheFolder = actionResult ? AC_DIRECTORY : CAS_DIRECTORY;
    // Create the file in a subfolder to bypass possible folder file count limits
    return root.getChild(cacheFolder).getChild(key.substring(0, 2)).getChild(key);
  }

  private void saveFile(String key, InputStream in, boolean actionResult) throws IOException {
    Path target = toPath(key, actionResult);
    if (target.exists()) {
      return;
    }
    target.getParentDirectory().createDirectoryAndParents();

    // Write a temporary file first, and then rename, to avoid data corruption in case of a crash.
    Path temp = toPathNoSplit(UUID.randomUUID().toString());

    try (FileOutputStream out = new FileOutputStream(temp.getPathFile())) {
      ByteStreams.copy(in, out);
      // Fsync temp before we rename it to avoid data loss in the case of machine
      // crashes (the OS may reorder the writes and the rename).
      out.getFD().sync();
    }
    temp.renameTo(target);
  }
}
