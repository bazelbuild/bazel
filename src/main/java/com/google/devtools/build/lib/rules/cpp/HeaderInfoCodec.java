// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.base.Throwables.throwIfInstanceOf;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.collect.ImmutableList;
import com.google.common.hash.Hashing;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSetStore.NestedSetStorageEndpoint;
import com.google.devtools.build.lib.rules.cpp.CcCompilationContext.HeaderInfo;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.util.ArrayList;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;

/**
 * A codec for {@link HeaderInfo} with intra-value memoization.
 *
 * <p>{@link HeaderInfo} recursively refers to other {@link HeaderInfo} instances that must be
 * serialized in memoized fashion to avoid quadratic storage costs.
 */
public final class HeaderInfoCodec implements ObjectCodec<HeaderInfo> {
  private final NestedSetStorageEndpoint storageEndpoint;

  /**
   * An executor that performs deserialization computations when bytes are fetched from storage.
   *
   * <p>This exists to avoid tying up an RPC callback thread.
   */
  private final Executor executor;

  // TODO(b/297857068): while this is only in test code, a ConcurrentHashMap is fine. In production,
  // we will need to carefully consider how entries in these maps are invalidated. This could be as
  // simple as wiping them between invocations.
  // TODO(b/297857068): in both of the maps below, it is possible to unwrap the futures after they
  // complete to reduce memory consumption.
  private final ConcurrentHashMap<ByteString, ListenableFuture<HeaderInfo>> fingerprintToValue =
      new ConcurrentHashMap<>();
  private final ConcurrentHashMap<HeaderInfo, ListenableFuture<ByteString>> referenceToFingerprint =
      new ConcurrentHashMap<>();

  /**
   * A parameter set to improve coverage in round-trip testing.
   *
   * <p>When false, completion of serialization populates the {@link #fingerprintToValue} entries,
   * which is used to look up values from fingerprints during deserialization. However, this means
   * that when round-tripping in test scenarios, the deserialization isn't really exercised as it
   * always hits the populated {@link #fingerprintToValue} entries. Setting this flag true disables
   * populating {@link #fingerprintToValue} from serialization so the deserialization code can be
   * exercised.
   */
  private final boolean exerciseDeserializationForTesting;

  public HeaderInfoCodec(
      NestedSetStorageEndpoint storageEndpoint,
      Executor executor,
      boolean exerciseDeserializationForTesting) {
    this.storageEndpoint = storageEndpoint;
    this.executor = executor;
    this.exerciseDeserializationForTesting = exerciseDeserializationForTesting;
  }

  @Override
  public Class<HeaderInfo> getEncodedClass() {
    return HeaderInfo.class;
  }

  @Override
  public void serialize(SerializationContext context, HeaderInfo obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    ByteString fingerprint;
    try {
      fingerprint = serializeToStorage(context, obj).get();
    } catch (InterruptedException e) {
      throw new InterruptedIOException();
    } catch (ExecutionException e) {
      Throwable cause = e.getCause();
      throwIfInstanceOf(cause, SerializationException.class);
      throwIfInstanceOf(cause, IOException.class);
      throw new SerializationException("Unexpected execution exception", cause);
    }
    codedOut.writeBytesNoTag(fingerprint);
  }

  @Override
  public HeaderInfo deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    ByteString fingerprint = codedIn.readBytes();
    try {
      return deserializeFromStorage(context, fingerprint).get();
    } catch (InterruptedException e) {
      throw new InterruptedIOException();
    } catch (ExecutionException e) {
      Throwable cause = e.getCause();
      throwIfInstanceOf(cause, SerializationException.class);
      throwIfInstanceOf(cause, IOException.class);
      throw new SerializationException("Unexpected execution exception", cause);
    }
  }

  /**
   * Serializes {@code headerInfo} to storage.
   *
   * <p>{@link HeaderInfo} is a recursive data structure that may include an entire DAG of {@link
   * HeaderInfo} values. This method is called recursively as subvalues occur.
   *
   * @return a unique fingerprint mapped to the {@link HeaderInfo} value
   */
  private ListenableFuture<ByteString> serializeToStorage(
      SerializationContext baseContext, HeaderInfo headerInfo)
      throws SerializationException, IOException {
    var settableFingerprint = SettableFuture.<ByteString>create();
    var previousFingerprint = referenceToFingerprint.putIfAbsent(headerInfo, settableFingerprint);
    if (previousFingerprint != null) {
      return previousFingerprint;
    }

    // The context must be created fresh each time to reset the memoization state. The result should
    // be deserializable on its own, without additional context.
    SerializationContext context = baseContext.getNewMemoizingContext();
    // Care must be taken to ensure the SettableFuture is actually set to avoid hanging elsewhere.
    boolean futureWasSet = false;
    try {
      var bytesOut = new ByteArrayOutputStream();
      var codedOut = CodedOutputStream.newInstance(bytesOut);
      context.serialize(headerInfo.headerModule, codedOut);
      context.serialize(headerInfo.picHeaderModule, codedOut);
      context.serialize(headerInfo.modularPublicHeaders, codedOut);
      context.serialize(headerInfo.modularPrivateHeaders, codedOut);
      context.serialize(headerInfo.textualHeaders, codedOut);
      context.serialize(headerInfo.separateModuleHeaders, codedOut);
      context.serialize(headerInfo.separateModule, codedOut);
      context.serialize(headerInfo.separatePicModule, codedOut);

      ImmutableList<HeaderInfo> deps = headerInfo.deps;
      codedOut.writeInt32NoTag(deps.size());

      // In both branches below, the future waits until the bytes are stored in the NestedSetStore.
      // It may be possible to achieve higher throughput by making the fingerprints available as
      // soon as possible, without waiting for storage to complete, but that would require
      // additional coordination to be built somewhere.
      if (deps.isEmpty()) {
        codedOut.flush();
        settableFingerprint.setFuture(storeByFingerprint(headerInfo, bytesOut.toByteArray()));
      } else {
        var depFutures = new ArrayList<ListenableFuture<ByteString>>(deps.size());
        for (HeaderInfo dep : deps) {
          depFutures.add(serializeToStorage(baseContext, dep));
        }
        settableFingerprint.setFuture(
            Futures.whenAllSucceed(depFutures)
                .callAsync(
                    () -> {
                      for (var depFuture : depFutures) {
                        codedOut.writeBytesNoTag(Futures.getDone(depFuture));
                      }
                      codedOut.flush();
                      return storeByFingerprint(headerInfo, bytesOut.toByteArray());
                    },
                    directExecutor()));
      }
      futureWasSet = true;
    } catch (SerializationException | IOException e) {
      settableFingerprint.setException(e);
      futureWasSet = true;
      throw e;
    } finally {
      if (!futureWasSet) {
        // Fails fast (instead of potentially causing a client to hang) if there's an unanticipated
        // runtime exception.
        settableFingerprint.setException(new IllegalStateException("The future was not set!"));
      }
    }

    return settableFingerprint;
  }

  private ListenableFuture<HeaderInfo> deserializeFromStorage(
      DeserializationContext context, ByteString fingerprint) throws IOException {
    var settableValue = SettableFuture.<HeaderInfo>create();
    var previousValue = fingerprintToValue.putIfAbsent(fingerprint, settableValue);
    if (previousValue != null) {
      return previousValue;
    }

    boolean futureWasSet = false;
    try {
      settableValue.setFuture(
          Futures.transformAsync(
              storageEndpoint.get(fingerprint),
              bytes -> deserializeBytes(context, fingerprint, bytes),
              executor));
      futureWasSet = true;
    } catch (IOException e) {
      settableValue.setException(e);
      futureWasSet = true;
      throw e;
    } finally {
      if (!futureWasSet) {
        // Fails fast (instead of potentially causing a client to hang) if there's an unanticipated
        // runtime exception.
        settableValue.setException(new IllegalStateException("The future was not set!"));
      }
    }

    return settableValue;
  }

  private ListenableFuture<HeaderInfo> deserializeBytes(
      DeserializationContext baseContext, ByteString fingerprint, byte[] bytes)
      throws SerializationException, IOException {
    var codedIn = CodedInputStream.newInstance(bytes);
    var context = baseContext.getNewMemoizingContext();

    DerivedArtifact headerModule = context.deserialize(codedIn);
    DerivedArtifact picHeaderModule = context.deserialize(codedIn);
    ImmutableList<Artifact> modularPublicHeaders = context.deserialize(codedIn);
    ImmutableList<Artifact> modularPrivateHeaders = context.deserialize(codedIn);
    ImmutableList<Artifact> textualHeaders = context.deserialize(codedIn);
    ImmutableList<Artifact> separateModuleHeaders = context.deserialize(codedIn);
    DerivedArtifact separateModule = context.deserialize(codedIn);
    DerivedArtifact separatePicModule = context.deserialize(codedIn);

    int depCount = codedIn.readInt32();
    if (depCount == 0) {
      var headerInfo =
          new HeaderInfo(
              headerModule,
              picHeaderModule,
              modularPublicHeaders,
              modularPrivateHeaders,
              textualHeaders,
              separateModuleHeaders,
              separateModule,
              separatePicModule,
              ImmutableList.of());
      referenceToFingerprint.put(headerInfo, immediateFuture(fingerprint));
      return immediateFuture(headerInfo);
    }

    var futureDeps = new ArrayList<ListenableFuture<HeaderInfo>>(depCount);
    for (int i = 0; i < depCount; ++i) {
      futureDeps.add(deserializeFromStorage(baseContext, codedIn.readBytes()));
    }
    return Futures.whenAllSucceed(futureDeps)
        .call(
            () -> {
              var deps = ImmutableList.<HeaderInfo>builderWithExpectedSize(depCount);
              for (var futureDep : futureDeps) {
                deps.add(Futures.getDone(futureDep));
              }
              var headerInfo =
                  new HeaderInfo(
                      headerModule,
                      picHeaderModule,
                      modularPublicHeaders,
                      modularPrivateHeaders,
                      textualHeaders,
                      separateModuleHeaders,
                      separateModule,
                      separatePicModule,
                      deps.build());
              referenceToFingerprint.put(headerInfo, immediateFuture(fingerprint));
              return headerInfo;
            },
            directExecutor());
  }

  private ListenableFuture<ByteString> storeByFingerprint(HeaderInfo info, byte[] bytes)
      throws IOException {
    // TODO(b/297857068): observe that the fingerprint value is not made available until after a
    // round trip to through the storage service. This means that it should be possible to replace
    // the locally generated fingerprint with a storage-service generated UUID.
    ByteString fingerprint = ByteString.copyFrom(Hashing.md5().hashBytes(bytes).asBytes());
    if (!exerciseDeserializationForTesting) {
      fingerprintToValue.putIfAbsent(fingerprint, immediateFuture(info));
    }
    return Futures.transform(
        storageEndpoint.put(fingerprint, bytes), unusedVoid -> fingerprint, directExecutor());
  }
}
