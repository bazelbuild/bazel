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

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.remoteexecution.v1test.Digest;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A {@link BuildEventArtifactUploader} backed by {@link ByteStreamUploader}.
 */
class ByteStreamBuildEventArtifactUploader implements BuildEventArtifactUploader {

  private final ByteStreamUploader uploader;
  private final String remoteServerInstanceName;

  ByteStreamBuildEventArtifactUploader(
      ByteStreamUploader uploader, String remoteServerName, @Nullable String remoteInstanceName) {
    this.uploader = Preconditions.checkNotNull(uploader);
    String remoteServerInstanceName = Preconditions.checkNotNull(remoteServerName);
    if (!Strings.isNullOrEmpty(remoteInstanceName)) {
      remoteServerInstanceName += "/" + remoteInstanceName;
    }
    this.remoteServerInstanceName = remoteServerInstanceName;
  }

  @Override
  public ListenableFuture<PathConverter> upload(Map<Path, LocalFile> files) {
    if (files.isEmpty()) {
      return Futures.immediateFuture(PathConverter.NO_CONVERSION);
    }
    List<ListenableFuture<PathDigestPair>> uploads = new ArrayList<>(files.size());
    try {
      for (Path file : files.keySet()) {
        Chunker chunker = new Chunker(file);
        Digest digest = chunker.digest();
        ListenableFuture<PathDigestPair> upload =
            Futures.transform(
                uploader.uploadBlobAsync(chunker),
                unused -> new PathDigestPair(file, digest),
                MoreExecutors.directExecutor());
        uploads.add(upload);
      }

      return Futures.transform(Futures.allAsList(uploads),
          (uploadsDone) -> new PathConverterImpl(remoteServerInstanceName, uploadsDone),
          MoreExecutors.directExecutor());
    } catch (IOException e) {
      return Futures.immediateFailedFuture(e);
    }
  }

  private static class PathConverterImpl implements PathConverter {

    private final String remoteServerInstanceName;
    private final Map<Path, Digest> pathToDigest;

    PathConverterImpl(String remoteServerInstanceName,
        List<PathDigestPair> uploads) {
      Preconditions.checkNotNull(uploads);
      this.remoteServerInstanceName = remoteServerInstanceName;
      pathToDigest = new HashMap<>(uploads.size());
      for (PathDigestPair pair : uploads) {
        pathToDigest.put(pair.getPath(), pair.getDigest());
      }
    }

    @Override
    public String apply(Path path) {
      Preconditions.checkNotNull(path);
      Digest digest = pathToDigest.get(path);
      if (digest == null) {
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
