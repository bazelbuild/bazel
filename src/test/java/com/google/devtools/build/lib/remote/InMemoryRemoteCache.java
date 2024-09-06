// Copyright 2021 The Bazel Authors. All rights reserved.
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

import static java.nio.charset.StandardCharsets.UTF_8;

import build.bazel.remote.execution.v2.Digest;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.InMemoryCacheClient;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.protobuf.ByteString;
import com.google.protobuf.Message;
import java.io.IOException;
import java.util.Map;

class InMemoryRemoteCache extends RemoteExecutionCache {

  InMemoryRemoteCache(
      Map<Digest, byte[]> casEntries, RemoteOptions options, DigestUtil digestUtil) {
    super(new InMemoryCacheClient(casEntries), /* diskCacheClient= */ null, options, digestUtil);
  }

  InMemoryRemoteCache(RemoteOptions options, DigestUtil digestUtil) {
    super(new InMemoryCacheClient(), /* diskCacheClient= */ null, options, digestUtil);
  }

  InMemoryRemoteCache(
      RemoteCacheClient cacheProtocol, RemoteOptions options, DigestUtil digestUtil) {
    super(cacheProtocol, /* diskCacheClient= */ null, options, digestUtil);
  }

  Digest addContents(RemoteActionExecutionContext context, String txt)
      throws IOException, InterruptedException {
    return addContents(context, txt.getBytes(UTF_8));
  }

  Digest addContents(RemoteActionExecutionContext context, byte[] bytes)
      throws IOException, InterruptedException {
    Digest digest = digestUtil.compute(bytes);
    Utils.getFromFuture(remoteCacheClient.uploadBlob(context, digest, ByteString.copyFrom(bytes)));
    return digest;
  }

  Digest addContents(RemoteActionExecutionContext context, Message m)
      throws IOException, InterruptedException {
    return addContents(context, m.toByteArray());
  }

  Digest addException(String txt, Exception e) {
    Digest digest = digestUtil.compute(txt.getBytes(UTF_8));
    ((InMemoryCacheClient) remoteCacheClient).addDownloadFailure(digest, e);
    return digest;
  }

  Digest addException(Message m, Exception e) {
    Digest digest = digestUtil.compute(m);
    ((InMemoryCacheClient) remoteCacheClient).addDownloadFailure(digest, e);
    return digest;
  }

  int getNumSuccessfulDownloads() {
    return ((InMemoryCacheClient) remoteCacheClient).getNumSuccessfulDownloads();
  }

  int getNumFailedDownloads() {
    return ((InMemoryCacheClient) remoteCacheClient).getNumFailedDownloads();
  }

  Map<Digest, Integer> getNumFindMissingDigests() {
    return ((InMemoryCacheClient) remoteCacheClient).getNumFindMissingDigests();
  }
}
