// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.skyframe.serialization.FutureHelpers.waitForDeserializationFuture;
import static com.google.devtools.build.lib.unsafe.UnsafeProvider.unsafe;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec.DeferredValue;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueStore.MissingFingerprintValueException;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/** Implementation that supports sharing of sub-objects between objects. */
final class SharedValueDeserializationContext extends MemoizingDeserializationContext {
  private final FingerprintValueService fingerprintValueService;

  @VisibleForTesting // private
  static SharedValueDeserializationContext createForTesting(
      ObjectCodecRegistry codecRegistry,
      ImmutableClassToInstanceMap<Object> dependencies,
      FingerprintValueService fingerprintValueService) {
    return new SharedValueDeserializationContext(
        codecRegistry, dependencies, fingerprintValueService);
  }

  /**
   * List of futures that must be resolved before this value is completely deserialized.
   *
   * <p>The synchronous {@link #deserialize(CodedInputStream)} overload includes waiting for these
   * futures to to be resolved. The top-level deserialization call typically uses that overload
   * while codec implementations use the ones in {@link AsyncDeserializationContext}.
   */
  @Nullable // Initialized lazily.
  private ArrayList<ListenableFuture<?>> readStatusFutures;

  /**
   * Tracks which {@link #readStatusFutures} were added, transitively, while deserializing a value.
   *
   * <p>When deserializing a value, {@link MemoizingDeserializationContext} calls {@link
   * #deserializeAndMaybeHandleDeferredValues}, then {@link #combineValueWithReadFutures}.
   *
   * <p>{@link #deserializeAndMaybeHandleDeferredValues} performs the following steps.
   *
   * <ul>
   *   <li>It notes the size of {@link #readStatusFutures} when it begins.
   *   <li>It initiates deserialization of the next value with a given {@link ObjectCodec}.
   *   <li>After that deserialization invocation completes, it sets {@code lastStartingReadCount} to
   *       the size it noted when it started.
   * </ul>
   *
   * <p>Consequently, the futures in {@link #readStatusFutures} with index greater than {@code
   * lastStartingReadCount} are ones added (transitively) by deserialization.
   *
   * <p>Next, {@link #combineValueWithReadFutures} uses {@code lastStartingReadCount} to determine
   * what futures were added and need to be complete before deserialization of the value is
   * complete.
   */
  private int lastStartingReadCount;

  private SharedValueDeserializationContext(
      ObjectCodecRegistry codecRegistry,
      ImmutableClassToInstanceMap<Object> dependencies,
      FingerprintValueService fingerprintValueService) {
    super(codecRegistry, dependencies);
    this.fingerprintValueService = fingerprintValueService;
  }

  static Object deserializeWithSharedValues(
      ObjectCodecRegistry codecRegistry,
      ImmutableClassToInstanceMap<Object> dependencies,
      FingerprintValueService fingerprintValueService,
      ByteString bytes)
      throws SerializationException {
    try {
      return ObjectCodecs.deserializeStreamFully(
          bytes.newCodedInput(),
          new SharedValueDeserializationContext(
              codecRegistry, dependencies, fingerprintValueService));
    } catch (SerializationException e) {
      Throwable cause = e.getCause();
      if (cause instanceof MissingFingerprintValueException) {
        // TODO: b/297857068 - eventually, callers of this should handle this by falling back on
        // local recomputation.
        throw new IllegalStateException("Not yet supported.", cause);
      }
      throw e;
    }
  }

  @Override
  public void deserialize(CodedInputStream codedIn, Object parent, long offset)
      throws IOException, SerializationException {
    Object result = processTagAndDeserialize(codedIn);
    if (result == null) {
      return;
    }

    if (!(result instanceof ListenableFuture)) {
      unsafe().putObject(parent, offset, result);
      return;
    }

    addReadStatusFuture(
        Futures.transform(
            (ListenableFuture<?>) result,
            value -> {
              unsafe().putObject(parent, offset, value);
              return null;
            },
            directExecutor()));
  }

  @Override
  public <T> void deserialize(CodedInputStream codedIn, T parent, FieldSetter<? super T> setter)
      throws IOException, SerializationException {
    Object result = processTagAndDeserialize(codedIn);
    if (result == null) {
      return;
    }

    if (!(result instanceof ListenableFuture)) {
      setter.set(parent, result);
      return;
    }

    addReadStatusFuture(
        Futures.transformAsync(
            (ListenableFuture<?>) result,
            value -> {
              try {
                setter.set(parent, value);
              } catch (SerializationException e) {
                return immediateFailedFuture(e);
              }
              return immediateVoidFuture();
            },
            directExecutor()));
  }

  @Override
  public void deserialize(CodedInputStream codedIn, Object parent, long offset, Runnable done)
      throws IOException, SerializationException {
    Object result = processTagAndDeserialize(codedIn);
    if (result == null) {
      done.run();
      return;
    }

    if (!(result instanceof ListenableFuture)) {
      unsafe().putObject(parent, offset, result);
      done.run();
      return;
    }

    addReadStatusFuture(
        Futures.transform(
            (ListenableFuture<?>) result,
            value -> {
              unsafe().putObject(parent, offset, value);
              done.run();
              return null;
            },
            directExecutor()));
  }

  @Override
  public <T> void getSharedValue(
      CodedInputStream codedIn,
      @Nullable Object distinguisher,
      DeferredObjectCodec<?> codec,
      T parent,
      FieldSetter<? super T> setter)
      throws IOException, SerializationException {
    ByteString fingerprint =
        ByteString.copyFrom(codedIn.readRawBytes(fingerprintValueService.fingerprintLength()));
    SettableFuture<Object> getOperation = SettableFuture.create();
    Object previous =
        fingerprintValueService.getOrClaimGetOperation(fingerprint, distinguisher, getOperation);
    if (previous != null) {
      // This object was previously requested. Discards `getOperation`.
      if (previous instanceof ListenableFuture) {
        @SuppressWarnings("unchecked")
        ListenableFuture<Object> futureValue = (ListenableFuture<Object>) previous;
        addReadStatusFuture(
            Futures.transformAsync(
                futureValue,
                value -> {
                  try {
                    setter.set(parent, value);
                  } catch (SerializationException e) {
                    return immediateFailedFuture(e);
                  }
                  return immediateVoidFuture();
                },
                directExecutor()));
        return;
      }
      setter.set(parent, previous);
      return;
    }

    // There is no previous result. Fetches the remote bytes and deserializes them.
    addReadStatusFuture(readValueForFingerprint(fingerprint, codec, parent, setter, getOperation));
  }

  private <T> ListenableFuture<Object> readValueForFingerprint(
      ByteString fingerprint,
      DeferredObjectCodec<?> codec,
      T parent,
      FieldSetter<? super T> setter,
      SettableFuture<Object> getOperation)
      throws IOException {
    try {
      ListenableFuture<Object> futureResult =
          Futures.transformAsync(
              fingerprintValueService.get(fingerprint),
              bytes -> {
                SharedValueDeserializationContext innerContext = getFreshContext();
                try {
                  DeferredValue<?> deferred =
                      codec.deserializeDeferred(innerContext, CodedInputStream.newInstance(bytes));
                  List<ListenableFuture<?>> innerReadStatusFutures = innerContext.readStatusFutures;
                  if (innerReadStatusFutures == null || innerReadStatusFutures.isEmpty()) {
                    Object result = deferred.call();
                    setter.set(parent, result);
                    return immediateFuture(result);
                  }
                  return Futures.whenAllSucceed(innerReadStatusFutures)
                      .call(
                          () -> {
                            Object result = deferred.call();
                            setter.set(parent, result);
                            return result;
                          },
                          directExecutor());
                } catch (SerializationException | IOException e) {
                  return immediateFailedFuture(e);
                }
              },
              // Switches to another executor to avoid performing serialization work on an an RPC
              // executor thread.
              fingerprintValueService.getExecutor());
      getOperation.setFuture(futureResult);
      return futureResult;
    } catch (IOException e) {
      getOperation.setException(e);
      throw e;
    } catch (RuntimeException | Error e) {
      // Avoids causing SettableFuture consumers to hang if when there are unexpected exceptions.
      getOperation.setException(e);
      throw e;
    }
  }

  @Override
  public SharedValueDeserializationContext getFreshContext() {
    return new SharedValueDeserializationContext(
        getRegistry(), getDependencies(), fingerprintValueService);
  }

  @Override
  Object makeSynchronous(Object obj) throws SerializationException {
    if (obj instanceof ListenableFuture) {
      return waitForDeserializationFuture((ListenableFuture<?>) obj);
    }
    return obj;
  }

  @Override
  Object deserializeAndMaybeHandleDeferredValues(ObjectCodec<?> codec, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int startingReadCount = readStatusFutures == null ? 0 : readStatusFutures.size();

    Object value;
    if (codec instanceof DeferredObjectCodec) {
      // On other analogous codepaths, `ObjectCodec.safeCast' is applied to the resulting value.
      // Not all codecs have this property, notably DynamicCodec, but DeferredObjectCodec's type
      // parameters guarantee type of the deserialized value.
      value = ((DeferredObjectCodec<?>) codec).deserializeDeferred(this, codedIn);
    } else {
      value = codec.safeCast(codec.deserialize(this, codedIn));
    }

    this.lastStartingReadCount = startingReadCount;
    return value;
  }

  @Override
  @SuppressWarnings("FutureReturnValueIgnored")
  Object combineValueWithReadFutures(Object value) {
    if (readStatusFutures == null) {
      return unwrapIfDeferredValue(value);
    }
    int length = readStatusFutures.size();
    if (length <= lastStartingReadCount) {
      return unwrapIfDeferredValue(value);
    }

    List<ListenableFuture<?>> futures = readStatusFutures.subList(lastStartingReadCount, length);
    Futures.FutureCombiner<?> combiner = Futures.whenAllSucceed(futures);
    futures.clear(); // clears this sublist from from `readStatusFutures`

    ListenableFuture<Object> futureValue;
    if (value instanceof DeferredValue) {
      @SuppressWarnings("unchecked")
      DeferredValue<Object> castValue = (DeferredValue<Object>) value;
      futureValue = combiner.call(castValue, directExecutor());
    } else {
      futureValue = combiner.call(() -> value, directExecutor());
    }
    readStatusFutures.add(futureValue);
    return futureValue;
  }

  private void addReadStatusFuture(ListenableFuture<?> readStatus) {
    if (readStatusFutures == null) {
      readStatusFutures = new ArrayList<>();
    }
    readStatusFutures.add(readStatus);
  }

  private static Object unwrapIfDeferredValue(Object value) {
    if (value instanceof DeferredValue) {
      @SuppressWarnings("unchecked")
      DeferredValue<Object> castValue = (DeferredValue<Object>) value;
      return castValue.call();
    }
    return value;
  }
}
