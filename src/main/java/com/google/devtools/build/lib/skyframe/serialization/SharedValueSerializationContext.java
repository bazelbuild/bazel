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

import static com.google.devtools.build.lib.skyframe.serialization.FutureHelpers.aggregateStatusFutures;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import javax.annotation.Nullable;

/**
 * A {@link SerializationContext} that supports both memoization and shared subobjects.
 *
 * <p>Sharing sub-objects means uploading them asynchronously to a backing store. The status of
 * these uploads may be observed through {@link #createFutureToBlockWritingOn} and the {@link
 * SerializationResult#getFutureToBlockWritesOn}.
 */
final class SharedValueSerializationContext extends MemoizingSerializationContext {
  /**
   * Futures that represent writes to remote storage.
   *
   * <p>For consistency, the serialized bytes should not be published for other consumers until
   * these writes complete.
   */
  @Nullable // lazily initialized
  private ArrayList<ListenableFuture<Void>> futuresToBlockWritingOn;

  @VisibleForTesting // private
  static SharedValueSerializationContext createForTesting(
      ObjectCodecRegistry codecRegistry, ImmutableClassToInstanceMap<Object> dependencies) {
    return new SharedValueSerializationContext(codecRegistry, dependencies);
  }

  private SharedValueSerializationContext(
      ObjectCodecRegistry codecRegistry, ImmutableClassToInstanceMap<Object> dependencies) {
    super(codecRegistry, dependencies);
  }

  /**
   * Serializes {@code subject} and returns a result that may have an associated future.
   *
   * <p>This method does not block on uploads. Instead, upload status is provided by {@link
   * SerializationResult#getFutureToBlockWritesOn}.
   */
  static SerializationResult<ByteString> serializeToResult(
      ObjectCodecRegistry codecRegistry,
      ImmutableClassToInstanceMap<Object> dependencies,
      @Nullable Object subject)
      throws SerializationException {
    SharedValueSerializationContext context =
        new SharedValueSerializationContext(codecRegistry, dependencies);
    ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(bytesOut);
    try {
      context.serialize(subject, codedOut);
      codedOut.flush();
    } catch (IOException e) {
      throw new SerializationException("Failed to serialize: " + subject, e);
    }
    return context.createResult(bytesOut.toByteArray());
  }

  @Override
  public SharedValueSerializationContext getFreshContext() {
    return new SharedValueSerializationContext(getCodecRegistry(), getDependencies());
  }

  /**
   * Registers a {@link ListenableFuture} that must complete successfully before the serialized
   * bytes generated using this context can be written remotely.
   */
  @Override
  public void addFutureToBlockWritingOn(ListenableFuture<Void> future) {
    if (futuresToBlockWritingOn == null) {
      futuresToBlockWritingOn = new ArrayList<>();
    }
    futuresToBlockWritingOn.add(future);
  }

  @Override
  @Nullable
  public ListenableFuture<Void> createFutureToBlockWritingOn() {
    if (futuresToBlockWritingOn == null) {
      return null;
    }
    return aggregateStatusFutures(futuresToBlockWritingOn);
  }

  private SerializationResult<ByteString> createResult(byte[] bytes) {
    // TODO: b/297857068 - If ByteString.copyFrom overhead is excessive, use reflection to avoid it.
    ByteString finalBytes = ByteString.copyFrom(bytes);
    return futuresToBlockWritingOn == null
        ? SerializationResult.createWithoutFuture(finalBytes)
        : SerializationResult.create(finalBytes, aggregateStatusFutures(futuresToBlockWritingOn));
  }
}
