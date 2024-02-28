// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ClassToInstanceMap;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.Maps;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * Wrapper for the minutiae of serializing and deserializing objects using {@link ObjectCodec}s,
 * serving as a layer between the streaming-oriented {@link ObjectCodec} interface and users.
 */
public class ObjectCodecs {
  /**
   * Default initial capacity of the output byte stream.
   *
   * <p>The same value that ByteArrayOutputStream's default constructor uses.
   */
  private static final int DEFAULT_OUTPUT_CAPACITY = 32;

  private final ImmutableSerializationContext serializationContext;
  private final DeserializationContext deserializationContext;

  /**
   * Creates an instance using the supplied {@link ObjectCodecRegistry} for looking up {@link
   * ObjectCodec}s.
   */
  public ObjectCodecs(
      ObjectCodecRegistry codecRegistry, ImmutableClassToInstanceMap<Object> dependencies) {
    serializationContext = new ImmutableSerializationContext(codecRegistry, dependencies);
    deserializationContext = new DeserializationContext(codecRegistry, dependencies);
  }

  public ObjectCodecs(ObjectCodecRegistry codecRegistry) {
    this(codecRegistry, ImmutableClassToInstanceMap.of());
  }

  public ObjectCodecs(ImmutableClassToInstanceMap<Object> dependencies) {
    this(AutoRegistry.get(), dependencies);
  }

  public ObjectCodecs() {
    this(AutoRegistry.get(), ImmutableClassToInstanceMap.of());
  }

  @Nullable
  public byte[] getCodecRegistryChecksum() {
    return getCodecRegistry().getChecksum();
  }

  @VisibleForTesting // private
  public ImmutableSerializationContext getSerializationContextForTesting() {
    return serializationContext;
  }

  @VisibleForTesting // private
  public MemoizingSerializationContext getMemoizingSerializationContextForTesting() {
    return MemoizingSerializationContext.createForTesting(getCodecRegistry(), getDependencies());
  }

  @VisibleForTesting // private
  public SharedValueSerializationContext getSharedValueSerializationContextForTesting() {
    return SharedValueSerializationContext.createForTesting(getCodecRegistry(), getDependencies());
  }

  @VisibleForTesting // private
  public ObjectCodecs withDependencyOverridesForTesting(ClassToInstanceMap<?> dependencyOverrides) {
    return new ObjectCodecs(
        getCodecRegistry(), overrideDependencies(getDependencies(), dependencyOverrides));
  }

  public DeserializationContext getDeserializationContext() {
    return deserializationContext;
  }

  /**
   * Serializes {@code obj} using a naive traversal.
   *
   * <p>This approach works well for simple, tree values. However, the naive traversal will stack
   * overflow on cyclic structures and can exhibit exponential complexity for DAGs.
   */
  public ByteString serialize(Object subject) throws SerializationException {
    ByteString.Output bytesOut = ByteString.newOutput();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(bytesOut);
    try {
      serializationContext.serialize(subject, codedOut);
      codedOut.flush();
    } catch (IOException e) {
      throw new SerializationException("Failed to serialize " + subject, e);
    }
    return bytesOut.toByteString();
  }

  public void serialize(Object subject, CodedOutputStream codedOut) throws SerializationException {
    try {
      serializationContext.serialize(subject, codedOut);
    } catch (IOException e) {
      throw new SerializationException("Failed to serialize " + subject, e);
    }
  }

  /** Serializes {@code subject} using memoization. */
  public ByteString serializeMemoized(@Nullable Object subject) throws SerializationException {
    return MemoizingSerializationContext.serializeToByteString(
        getCodecRegistry(),
        getDependencies(),
        subject,
        DEFAULT_OUTPUT_CAPACITY,
        CodedOutputStream.DEFAULT_BUFFER_SIZE);
  }

  /**
   * Serializes {@code subject} using memoization, with {@code byte[]} output.
   *
   * @param outputCapacity the initial capacity of the {@link ByteArrayOutputStream}
   * @param bufferSize size passed to {@link CodedOutputStream#newInstance}
   */
  public byte[] serializeMemoizedToBytes(
      @Nullable Object subject, int outputCapacity, int bufferSize) throws SerializationException {
    return MemoizingSerializationContext.serializeToBytes(
        getCodecRegistry(), getDependencies(), subject, outputCapacity, bufferSize);
  }

  /** Serializes {@code subject} using a {@link SharedValueSerializationContext}. */
  public SerializationResult<ByteString> serializeMemoizedAndBlocking(Object subject)
      throws SerializationException {
    return SharedValueSerializationContext.serializeToResult(
        getCodecRegistry(), getDependencies(), subject);
  }

  /**
   * Serializes {@code subject} using a {@link SharedValueSerializationContext}.
   *
   * @param dependencyOverrides dependencies to override, see {@link #overrideDependencies}
   */
  public SerializationResult<ByteString> serializeMemoizedAndBlocking(
      Object subject, ClassToInstanceMap<?> dependencyOverrides) throws SerializationException {
    return SharedValueSerializationContext.serializeToResult(
        getCodecRegistry(), overrideDependencies(getDependencies(), dependencyOverrides), subject);
  }

  public static Object deserialize(CodedInputStream codedIn, DeserializationContext context)
      throws SerializationException {
    // Allows access to buffer without copying (although this means buffer may be pinned in memory).
    codedIn.enableAliasing(true);
    Object result;
    try {
      result = context.deserialize(codedIn);
    } catch (IOException e) {
      throw new SerializationException("Failed to deserialize data", e);
    }
    try {
      if (!codedIn.isAtEnd()) {
        throw new SerializationException(
            "input stream not exhausted after deserializing " + result);
      }
    } catch (IOException e) {
      throw new SerializationException("Error checking for end of stream with " + result, e);
    }
    return result;
  }

  public Object deserialize(ByteString data) throws SerializationException {
    return deserialize(data.newCodedInput());
  }

  public Object deserialize(CodedInputStream codedIn) throws SerializationException {
    return deserialize(codedIn, deserializationContext);
  }

  public Object deserializeMemoized(ByteString data) throws SerializationException {
    return deserializeMemoized(data.newCodedInput());
  }

  public Object deserializeMemoized(
      ByteString data, ImmutableClassToInstanceMap<?> dependencyOverrides)
      throws SerializationException {
    return deserialize(
        data.newCodedInput(),
        deserializationContext.withDependencyOverrides(dependencyOverrides).getMemoizingContext());
  }

  public Object deserializeMemoized(CodedInputStream codedIn) throws SerializationException {
    return deserialize(codedIn, deserializationContext.getMemoizingContext());
  }

  // It's awkward that values are read from `serializationContext` instead of
  // `deserializationContext` and that they always have the same values. There's not much cohesion
  // between these two, however, so introducing an extra layer of indirection to store a (codec
  // registry, dependencies) tuple doesn't appear to be worth it.

  private ObjectCodecRegistry getCodecRegistry() {
    return serializationContext.getCodecRegistry();
  }

  private ImmutableClassToInstanceMap<Object> getDependencies() {
    return serializationContext.getDependencies();
  }

  /**
   * Returns a new dependency map composed by applying overrides to {@code dependencies}.
   *
   * <p>The given {@code dependencyOverrides} may contain keys already present (in which case the
   * dependency will be replaced) or new keys (in which case the dependency will be added).
   */
  private static ImmutableClassToInstanceMap<Object> overrideDependencies(
      ImmutableClassToInstanceMap<Object> dependencies, ClassToInstanceMap<?> dependencyOverrides) {
    return ImmutableClassToInstanceMap.builder()
        .putAll(Maps.filterKeys(dependencies, k -> !dependencyOverrides.containsKey(k)))
        .putAll(dependencyOverrides)
        .build();
  }
}
