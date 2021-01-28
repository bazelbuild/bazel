// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.util;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Digest;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.ByteStreams;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;

/** A {@link RemoteCache} that stores its contents in memory. */
public class InMemoryCacheClient implements RemoteCacheClient {

  private final ConcurrentMap<Digest, Exception> downloadFailures = new ConcurrentHashMap<>();
  private final ConcurrentMap<ActionKey, ActionResult> ac = new ConcurrentHashMap<>();
  private final ConcurrentMap<Digest, byte[]> cas;

  private AtomicInteger numSuccess = new AtomicInteger();
  private AtomicInteger numFailures = new AtomicInteger();

  public InMemoryCacheClient(Map<Digest, byte[]> casEntries) {
    this.cas = new ConcurrentHashMap<>();
    for (Map.Entry<Digest, byte[]> entry : casEntries.entrySet()) {
      cas.put(entry.getKey(), entry.getValue());
    }
  }

  public InMemoryCacheClient() {
    this.cas = new ConcurrentHashMap<>();
  }

  public void addDownloadFailure(Digest digest, Exception e) {
    downloadFailures.put(digest, e);
  }

  public int getNumSuccessfulDownloads() {
    return numSuccess.get();
  }

  public int getNumFailedDownloads() {
    return numFailures.get();
  }

  @Override
  public ListenableFuture<Void> downloadBlob(Digest digest, OutputStream out) {
    Exception failure = downloadFailures.get(digest);
    if (failure != null) {
      numFailures.incrementAndGet();
      return Futures.immediateFailedFuture(failure);
    }

    byte[] data = cas.get(digest);
    if (data == null) {
      return Futures.immediateFailedFuture(new CacheNotFoundException(digest));
    }

    try {
      out.write(data);
      out.flush();
    } catch (IOException e) {
      numFailures.incrementAndGet();
      return Futures.immediateFailedFuture(e);
    }
    numSuccess.incrementAndGet();
    return Futures.immediateFuture(null);
  }

  @Override
  public ListenableFuture<ActionResult> downloadActionResult(
      RemoteActionExecutionContext context, ActionKey actionKey, boolean inlineOutErr) {
    ActionResult actionResult = ac.get(actionKey);
    if (actionResult == null) {
      return Futures.immediateFailedFuture(new CacheNotFoundException(actionKey.getDigest()));
    }
    return Futures.immediateFuture(actionResult);
  }

  @Override
  public void uploadActionResult(
      RemoteActionExecutionContext context, ActionKey actionKey, ActionResult actionResult) {
    ac.put(actionKey, actionResult);
  }

  @Override
  public ListenableFuture<Void> uploadFile(Digest digest, Path file) {
    try (InputStream in = file.getInputStream()) {
      cas.put(digest, ByteStreams.toByteArray(in));
    } catch (IOException e) {
      return Futures.immediateFailedFuture(e);
    }
    return Futures.immediateFuture(null);
  }

  @Override
  public ListenableFuture<Void> uploadBlob(Digest digest, ByteString data) {
    try (InputStream in = data.newInput()) {
      cas.put(digest, data.toByteArray());
    } catch (IOException e) {
      return Futures.immediateFailedFuture(e);
    }
    return Futures.immediateFuture(null);
  }

  @Override
  public ListenableFuture<ImmutableSet<Digest>> findMissingDigests(Iterable<Digest> digests) {
    ImmutableSet.Builder<Digest> missingBuilder = ImmutableSet.builder();
    for (Digest digest : digests) {
      if (!cas.containsKey(digest)) {
        missingBuilder.add(digest);
      }
    }
    return Futures.immediateFuture(missingBuilder.build());
  }

  @Override
  public void close() {
    cas.clear();
    ac.clear();
  }
}
