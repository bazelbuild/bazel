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
import com.google.common.hash.HashCode;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import io.grpc.Context;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Collection;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;
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
  private final String remoteInstanceName;
  private final CasDigestLookup digestLookup;

  ByteStreamBuildEventArtifactUploader(
      ByteStreamUploader uploader,
      String remoteServerName,
      Context ctx,
      @Nullable String remoteInstanceName,
      int maxUploadThreads,
      CasDigestLookup digestLookup) {
    this.uploader = Preconditions.checkNotNull(uploader);
    String remoteServerInstanceName = Preconditions.checkNotNull(remoteServerName);
    this.remoteInstanceName = remoteInstanceName;
    if (!Strings.isNullOrEmpty(this.remoteInstanceName)) {
      remoteServerInstanceName += "/" + this.remoteInstanceName;
    }
    this.ctx = ctx;
    this.remoteServerInstanceName = remoteServerInstanceName;
    this.digestLookup = digestLookup;
    // Limit the maximum threads number to 1000 (chosen arbitrarily)
    this.uploadExecutor =
        MoreExecutors.listeningDecorator(
            Executors.newFixedThreadPool(Math.min(maxUploadThreads, 1000)));
  }

  private static boolean isRemoteFile(Path file) {
    return file.getFileSystem() instanceof RemoteActionFileSystem
        && ((RemoteActionFileSystem) file.getFileSystem()).isRemote(file);
  }

  @Override
  public ListenableFuture<PathConverter> upload(Map<Path, LocalFile> files) {
    if (files.isEmpty()) {
      return Futures.immediateFuture(PathConverter.NO_CONVERSION);
    }

    // create futures for non-uploads:
    //  - directories should not be uploaded to CAS, and have no path
    //  - remote files should not be uploaded to CAS, but have a path
    List<ListenableFuture<Pair<Path, PathDigestPair>>> futures = files.keySet().stream()
        .map(file -> Futures.transformAsync(
            uploadExecutor.submit(() -> file.isDirectory()),
            isDirectory -> {
              Pair<Path, PathDigestPair> pair;
              if (isDirectory) {
                pair = new Pair<>(file, new PathDigestPair(file, null));
              } else if (isRemoteFile(file) || !files.get(file).shouldUploadArtifacts) {
                DigestUtil digestUtil = new DigestUtil(file.getFileSystem().getDigestFunction());
                pair = new Pair<>(file, new PathDigestPair(file, digestUtil.compute(file)));
              } else {
                pair = new Pair<>(file, null); // we potentially need to upload this
              }
              return Futures.immediateFuture(pair);
            },
            MoreExecutors.directExecutor()
        ))
        .collect(Collectors.toList());

    // perform missing blob check and upload missing items
    ListenableFuture<List<PathDigestPair>> pathDigestPairFuture = Futures.transformAsync(
        Futures.allAsList(futures),
        this::uploadMissingFiles,
        MoreExecutors.directExecutor()
    );

    return Futures.transform(
        pathDigestPairFuture,
        pathDigestPairs -> new PathConverterImpl(remoteServerInstanceName, pathDigestPairs),
        MoreExecutors.directExecutor());
  }

  private ListenableFuture<List<PathDigestPair>> uploadMissingFiles(List<Pair<Path, PathDigestPair>> pairs)
      throws IOException, InterruptedException {
    List<ListenableFuture<PathDigestPair>> pathDigestFutures = new ArrayList<>(pairs.size());
    Map<Digest, List<Path>> toUpload = new HashMap<>();

    // mark all entries without a PathDigestPair for upload
    for (Pair<Path, PathDigestPair> pathPair : pairs) {
      Path path = pathPair.first;
      if (pathPair.second == null) {
        DigestUtil digestUtil = new DigestUtil(path.getFileSystem().getDigestFunction());
        Digest digest = digestUtil.compute(path);
        if (!toUpload.containsKey(digest)) {
          toUpload.put(digest, new LinkedList<>());
        }
        toUpload.get(digest).add(path);
      } else {
        pathDigestFutures.add(Futures.immediateFuture(pathPair.second));
      }
    }

    Context prevContext = ctx.attach();
    ImmutableSet<Digest> missingDigests = digestLookup.getMissingDigests(toUpload.keySet());
    ctx.detach(prevContext);

    // new hashset because removeAll mutates the map
    Set<Digest> cachedDigests = new HashSet<>(toUpload.keySet());
    cachedDigests.removeAll(missingDigests);

    List<ListenableFuture<PathDigestPair>> cachedPairFutures = new ArrayList<>();

    // if not on the CAS, add the result of the upload
    for (Digest digest : missingDigests) {
      for (Path path : toUpload.get(digest)) {
        cachedPairFutures.add(uploadFile(path, digest));
      }
    }

    // otherwise, if cached, just add the file
    for (Digest digest : cachedDigests) {
      for (Path path : toUpload.get(digest)) {
        cachedPairFutures.add(Futures.immediateFuture(new PathDigestPair(path, digest)));
      }
    }

    return Futures.transform(
        Futures.allAsList(
            Futures.allAsList(pathDigestFutures), Futures.allAsList(cachedPairFutures)),
        futureLists -> futureLists.stream()
            .flatMap(Collection::stream)
            .collect(Collectors.toList()),
        MoreExecutors.directExecutor()
    );
  }

  private ListenableFuture<PathDigestPair> uploadFile(Path file, Digest digest) {
    Chunker chunker = Chunker.builder().setInput(digest.getSizeBytes(), file).build();
    final ListenableFuture<Void> upload;
    Context prevCtx = ctx.attach();
    try {
      upload =
          uploader.uploadBlobAsync(
              HashCode.fromString(digest.getHash()), chunker, /* forceUpload=*/ false);
    } finally {
      ctx.detach(prevCtx);
    }

    return Futures.transform(upload, unused -> new PathDigestPair(file, digest), uploadExecutor);
  }

  @Override
  public boolean mayBeSlow() {
    return true;
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
