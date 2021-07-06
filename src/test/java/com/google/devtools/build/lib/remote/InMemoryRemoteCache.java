package com.google.devtools.build.lib.remote;

import static java.nio.charset.StandardCharsets.UTF_8;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.InMemoryCacheClient;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.protobuf.ByteString;
import com.google.protobuf.Message;
import java.io.IOException;
import java.util.Map;

class InMemoryRemoteCache extends RemoteCache {

  InMemoryRemoteCache(
      Map<Digest, byte[]> casEntries, RemoteOptions options, DigestUtil digestUtil) {
    super(new InMemoryCacheClient(casEntries), options, digestUtil);
  }

  InMemoryRemoteCache(RemoteOptions options, DigestUtil digestUtil) {
    super(new InMemoryCacheClient(), options, digestUtil);
  }

  Digest addContents(RemoteActionExecutionContext context, String txt)
      throws IOException, InterruptedException {
    return addContents(context, txt.getBytes(UTF_8));
  }

  Digest addContents(RemoteActionExecutionContext context, byte[] bytes)
      throws IOException, InterruptedException {
    Digest digest = digestUtil.compute(bytes);
    Utils.getFromFuture(cacheProtocol.uploadBlob(context, digest, ByteString.copyFrom(bytes)));
    return digest;
  }

  Digest addContents(RemoteActionExecutionContext context, Message m)
      throws IOException, InterruptedException {
    return addContents(context, m.toByteArray());
  }

  Digest addException(String txt, Exception e) {
    Digest digest = digestUtil.compute(txt.getBytes(UTF_8));
    ((InMemoryCacheClient) cacheProtocol).addDownloadFailure(digest, e);
    return digest;
  }

  Digest addException(Message m, Exception e) {
    Digest digest = digestUtil.compute(m);
    ((InMemoryCacheClient) cacheProtocol).addDownloadFailure(digest, e);
    return digest;
  }

  int getNumSuccessfulDownloads() {
    return ((InMemoryCacheClient) cacheProtocol).getNumSuccessfulDownloads();
  }

  int getNumFailedDownloads() {
    return ((InMemoryCacheClient) cacheProtocol).getNumFailedDownloads();
  }

  ImmutableSet<Digest> findMissingDigests(
      RemoteActionExecutionContext context, Iterable<Digest> digests)
      throws IOException, InterruptedException {
    return Utils.getFromFuture(cacheProtocol.findMissingDigests(context, digests));
  }

  @Override
  public void close() {
    cacheProtocol.close();
  }
}
