// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.remote;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.Path;
import io.grpc.Context;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

/**
 * A {@link BuildEventArtifactUploader} backed by {@link ByteStreamUploader}.
 */
class ByteStreamBuildEventArtifactUploader implements BuildEventArtifactUploader {

  private final ListeningExecutorService uploadExecutor;
  private final Context ctx;
  private final ByteStreamUploader uploader;
  private final String remoteServerInstanceName;

  private final AtomicBoolean shutdown = new AtomicBoolean();

  ByteStreamBuildEventArtifactUploader(
      ByteStreamUploader uploader,
      String remoteServerName,
      Context ctx,
      @Nullable String remoteInstanceName,
      int maxUploadThreads) {
    this.uploader = Preconditions.checkNotNull(uploader);
    String remoteServerInstanceName = Preconditions.checkNotNull(remoteServerName);
    if (!Strings.isNullOrEmpty(remoteInstanceName)) {
      remoteServerInstanceName += "/" + remoteInstanceName;
    }
    this.ctx = ctx;
    this.remoteServerInstanceName = remoteServerInstanceName;
    // Limit the maximum threads number to 1000 (chosen arbitrarily)
    this.uploadExecutor =
        MoreExecutors.listeningDecorator(
            Executors.newFixedThreadPool(Math.min(maxUploadThreads, 1000)));
  }

  @Override
  public ListenableFuture<PathConverter> upload(Map<Path, LocalFile> files) {
    if (files.isEmpty()) {
      return Futures.immediateFuture(PathConverter.NO_CONVERSION);
    }
    List<ListenableFuture<PathDigestPair>> uploads = new ArrayList<>(files.size());

      for (Path file : files.keySet()) {
      ListenableFuture<Boolean> isDirectoryFuture = uploadExecutor.submit(() -> file.isDirectory());
      ListenableFuture<PathDigestPair> digestFuture =
          Futures.transformAsync(
              isDirectoryFuture,
              isDirectory -> {
                if (isDirectory) {
                  return Futures.immediateFuture(new PathDigestPair(file, null));
                }
                DigestUtil digestUtil = new DigestUtil(file.getFileSystem().getDigestFunction());
                Digest digest = digestUtil.compute(file);
                Chunker chunker = Chunker.builder(digestUtil).setInput(digest, file).build();
                final ListenableFuture<Void> upload;
                Context prevCtx = ctx.attach();
                try {
                  upload = uploader.uploadBlobAsync(chunker, /*forceUpload=*/ false);
                } finally {
                  ctx.detach(prevCtx);
                }
                return Futures.transform(
                    upload, unused -> new PathDigestPair(file, digest), uploadExecutor);
              },
              MoreExecutors.directExecutor());
      uploads.add(digestFuture);
      }
    return Futures.transform(
        Futures.allAsList(uploads),
        pathDigestPairs -> new PathConverterImpl(remoteServerInstanceName, pathDigestPairs),
        MoreExecutors.directExecutor());
  }

  @Override
  public void shutdown() {
    if (shutdown.getAndSet(true)) {
      return;
    }
    uploader.release();
  }

  private static class PathConverterImpl implements PathConverter {

    private final String remoteServerInstanceName;
    private final Map<Path, Digest> pathToDigest;
    private final Set<Path> skippedPaths;

    PathConverterImpl(String remoteServerInstanceName, List<PathDigestPair> uploads) {
      Preconditions.checkNotNull(uploads);
      this.remoteServerInstanceName = remoteServerInstanceName;
      pathToDigest = new HashMap<>(uploads.size());
      ImmutableSet.Builder<Path> skippedPaths = ImmutableSet.builder();
      for (PathDigestPair pair : uploads) {
        Path path = pair.getPath();
        Digest digest = pair.getDigest();
        if (digest != null) {
          pathToDigest.put(path, digest);
        } else {
          skippedPaths.add(path);
        }
      }
      this.skippedPaths = skippedPaths.build();
    }

    @Override
    public String apply(Path path) {
      Preconditions.checkNotNull(path);
      Digest digest = pathToDigest.get(path);
      if (digest == null) {
        if (skippedPaths.contains(path)) {
          return null;
        }
        // It's a programming error to reference a file that has not been uploaded.
        throw new IllegalStateException(
            String.format("Illegal file reference: '%s'", path.getPathString()));
      }
      return String.format(
          "bytestream://%s/blobs/%s/%d",
          remoteServerInstanceName, digest.getHash(), digest.getSizeBytes());
    }
  }

  private static class PathDigestPair {

    private final Path path;
    private final Digest digest;

    PathDigestPair(Path path, Digest digest) {
      this.path = path;
      this.digest = digest;
    }

    public Path getPath() {
      return path;
    }

    public Digest getDigest() {
      return digest;
    }
  }
}
