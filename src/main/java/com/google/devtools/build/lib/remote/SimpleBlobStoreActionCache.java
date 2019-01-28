// Copyright 2016 The Bazel Authors. All rights reserved.
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

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.FileNode;
import com.google.common.hash.HashingOutputStream;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.remote.blobstore.SimpleBlobStore;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.DigestUtil.ActionKey;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Collection;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/**
 * A RemoteActionCache implementation that uses a simple blob store for files and action output.
 *
 * <p>The thread safety is guaranteed by the underlying simple blob store.
 *
 * <p>Note that this class is used from src/tools/remote.
 */
@ThreadSafe
public final class SimpleBlobStoreActionCache extends AbstractRemoteActionCache {
  private static final int MAX_BLOB_SIZE_FOR_INLINE = 10 * 1024;

  private final SimpleBlobStore blobStore;

  private final ConcurrentHashMap<String, Boolean> storedBlobs;

  public SimpleBlobStoreActionCache(
      RemoteOptions options, SimpleBlobStore blobStore, Retrier retrier, DigestUtil digestUtil) {
    super(options, digestUtil, retrier);
    this.blobStore = blobStore;
    this.storedBlobs = new ConcurrentHashMap<>();
  }

  public void downloadTree(Digest rootDigest, Path rootLocation)
      throws IOException, InterruptedException {
    rootLocation.createDirectoryAndParents();
    Directory directory = Directory.parseFrom(getFromFuture(downloadBlob(rootDigest)));
    for (FileNode file : directory.getFilesList()) {
      Path dst = rootLocation.getRelative(file.getName());
      getFromFuture(downloadFile(dst, file.getDigest()));
      dst.setExecutable(file.getIsExecutable());
    }
    for (DirectoryNode child : directory.getDirectoriesList()) {
      downloadTree(child.getDigest(), rootLocation.getRelative(child.getName()));
    }
  }

  private Digest uploadFileContents(Path file) throws IOException, InterruptedException {
    Digest digest = digestUtil.compute(file);
    try (InputStream in = file.getInputStream()) {
      return uploadStream(digest, in);
    }
  }

  @Override
  public void upload(
      DigestUtil.ActionKey actionKey,
      Action action,
      Command command,
      Path execRoot,
      Collection<Path> files,
      FileOutErr outErr)
      throws ExecException, IOException, InterruptedException {
    ActionResult.Builder result = ActionResult.newBuilder();
    upload(result, actionKey, action, command, execRoot, files, /* uploadAction= */ true);
    if (outErr.getErrorPath().exists()) {
      Digest stderr = uploadFileContents(outErr.getErrorPath());
      result.setStderrDigest(stderr);
    }
    if (outErr.getOutputPath().exists()) {
      Digest stdout = uploadFileContents(outErr.getOutputPath());
      result.setStdoutDigest(stdout);
    }
    blobStore.putActionResult(actionKey.getDigest().getHash(), result.build().toByteArray());
  }

  public void upload(
      ActionResult.Builder result,
      DigestUtil.ActionKey actionKey,
      Action action,
      Command command,
      Path execRoot,
      Collection<Path> files,
      boolean uploadAction)
      throws ExecException, IOException, InterruptedException {
    UploadManifest manifest =
        new UploadManifest(
            digestUtil,
            result,
            execRoot,
            options.incompatibleRemoteSymlinks,
            options.allowSymlinkUpload);
    manifest.addFiles(files);
    if (uploadAction) {
      manifest.addAction(actionKey, action, command);
    }

    for (Map.Entry<Digest, Path> entry : manifest.getDigestToFile().entrySet()) {
      try (InputStream in = entry.getValue().getInputStream()) {
        uploadStream(entry.getKey(), in);
      }
    }

    for (Map.Entry<Digest, Chunker> entry : manifest.getDigestToChunkers().entrySet()) {
      uploadBlob(entry.getValue().next().getData().toByteArray(), entry.getKey());
    }
  }

  public void uploadOutErr(ActionResult.Builder result, byte[] stdout, byte[] stderr)
      throws IOException, InterruptedException {
    if (stdout.length <= MAX_BLOB_SIZE_FOR_INLINE) {
      result.setStdoutRaw(ByteString.copyFrom(stdout));
    } else if (stdout.length > 0) {
      result.setStdoutDigest(uploadBlob(stdout));
    }
    if (stderr.length <= MAX_BLOB_SIZE_FOR_INLINE) {
      result.setStderrRaw(ByteString.copyFrom(stderr));
    } else if (stderr.length > 0) {
      result.setStderrDigest(uploadBlob(stderr));
    }
  }

  public Digest uploadBlob(byte[] blob) throws IOException, InterruptedException {
    return uploadBlob(blob, digestUtil.compute(blob));
  }

  private Digest uploadBlob(byte[] blob, Digest digest) throws IOException, InterruptedException {
    try (InputStream in = new ByteArrayInputStream(blob)) {
      return uploadStream(digest, in);
    }
  }

  public Digest uploadStream(Digest digest, InputStream in)
      throws IOException, InterruptedException {
    final String hash = digest.getHash();

    if (storedBlobs.putIfAbsent(hash, true) == null) {
      try {
        blobStore.put(hash, digest.getSizeBytes(), in);
      } catch (Exception e) {
        storedBlobs.remove(hash);
        throw e;
      }
    }

    return digest;
  }

  public boolean containsKey(Digest digest) throws IOException, InterruptedException {
    return blobStore.containsKey(digest.getHash());
  }

  @Override
  public ActionResult getCachedActionResult(ActionKey actionKey)
      throws IOException, InterruptedException {
    try {
      byte[] data = downloadActionResult(actionKey.getDigest());
      return ActionResult.parseFrom(data);
    } catch (InvalidProtocolBufferException | CacheNotFoundException e) {
      return null;
    }
  }

  private byte[] downloadActionResult(Digest digest) throws IOException, InterruptedException {
    if (digest.getSizeBytes() == 0) {
      return new byte[0];
    }
    // This unconditionally downloads the whole blob into memory!
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    boolean success = blobStore.getActionResult(digest.getHash(), out);
    if (!success) {
      throw new CacheNotFoundException(digest, digestUtil);
    }
    return out.toByteArray();
  }

  public void setCachedActionResult(ActionKey actionKey, ActionResult result)
      throws IOException, InterruptedException {
    blobStore.putActionResult(actionKey.getDigest().getHash(), result.toByteArray());
  }

  @Override
  public void close() {
    blobStore.close();
  }

  @Override
  protected ListenableFuture<Void> downloadBlob(Digest digest, OutputStream out) {
    SettableFuture<Void> outerF = SettableFuture.create();
    @Nullable
    HashingOutputStream hashOut =
        options.remoteVerifyDownloads ? digestUtil.newHashingOutputStream(out) : null;
    Futures.addCallback(
        blobStore.get(digest.getHash(), hashOut != null ? hashOut : out),
        new FutureCallback<Boolean>() {
          @Override
          public void onSuccess(Boolean found) {
            if (found) {
              try {
                if (hashOut != null) {
                  verifyContents(digest, hashOut);
                }
                out.flush();
                outerF.set(null);
              } catch (IOException e) {
                outerF.setException(e);
              }
            } else {
              outerF.setException(new CacheNotFoundException(digest, digestUtil));
            }
          }

          @Override
          public void onFailure(Throwable throwable) {
            outerF.setException(throwable);
          }
        },
        MoreExecutors.directExecutor());
    return outerF;
  }
}
