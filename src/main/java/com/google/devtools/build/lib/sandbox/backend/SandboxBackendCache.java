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

import build.bazel.remote.execution.v2.Digest;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.remote.ChunkLocationMap;
import com.google.devtools.build.lib.remote.RemoteExecutionCache;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.Blob;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import java.io.InputStream;
import java.util.OptionalLong;

/**
 * The uploader of the inputs of an action to a sandbox backend.
 *
 * <p>Everything but the contents of input files is uploaded as is. For an input file, only its
 * location is uploaded, see {@link LocationBlobs}, so that the backend can bind the digest to the
 * file without Bazel reading it.
 */
final class SandboxBackendCache extends RemoteExecutionCache {
  private final RemoteCacheClient client;

  SandboxBackendCache(RemoteCacheClient client, DigestUtil digestUtil) {
    super(
        client,
        /* diskCacheClient= */ null,
        /* symlinkTemplate= */ null,
        digestUtil,
        /* chunkingFunction= */ null,
        new ChunkLocationMap());
    this.client = client;
  }

  @Override
  public ListenableFuture<Void> uploadFile(
      RemoteActionExecutionContext context,
      Digest digest,
      Path path,
      PathFragment execPath,
      boolean force) {
    // All inputs have been prefetched, so the path is a path on the host file system.
    return client.uploadBlob(context, digest, new LocationBlob(path), force);
  }

  private record LocationBlob(Path path) implements Blob {
    @Override
    public InputStream get() {
      return location().newInput();
    }

    @Override
    public String description() {
      return "location of file " + path;
    }

    @Override
    public OptionalLong streamedSize() {
      return OptionalLong.of(location().size());
    }

    private ByteString location() {
      return LocationBlobs.encode(path.getPathString());
    }
  }
}
