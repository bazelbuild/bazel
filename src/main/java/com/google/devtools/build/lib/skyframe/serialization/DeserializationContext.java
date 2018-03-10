// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skyframe.serialization.Memoizer.Deserializer;
import com.google.devtools.build.lib.skyframe.serialization.Memoizer.MemoizingAfterObjectCodecAdapter;
import com.google.devtools.build.lib.skyframe.serialization.Memoizer.MemoizingCodec;
import com.google.devtools.build.lib.skyframe.serialization.Memoizer.Serializer;
import com.google.devtools.build.lib.skyframe.serialization.Memoizer.Strategy;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import javax.annotation.Nullable;

/** Stateful class for providing additional context to a single deserialization "session". */
// TODO(bazel-team): This class is just a shell, fill in.
public class DeserializationContext {

  private final ObjectCodecRegistry registry;
  private final ImmutableMap<Class<?>, Object> dependencies;

  public DeserializationContext(
      ObjectCodecRegistry registry, ImmutableMap<Class<?>, Object> dependencies) {
    this.registry = registry;
    this.dependencies = dependencies;
  }

  public DeserializationContext(ImmutableMap<Class<?>, Object> dependencies) {
    this(AutoRegistry.get(), dependencies);
  }

  // TODO(shahan): consider making codedIn a member of this class.
  @SuppressWarnings({"TypeParameterUnusedInFormals", "unchecked"})
  public <T> T deserialize(CodedInputStream codedIn) throws IOException, SerializationException {
    int tag = codedIn.readSInt32();
    if (tag == 0) {
      return null;
    }
    T constant = (T) registry.maybeGetConstantByTag(tag);
    return constant == null
        ? (T) registry.getCodecDescriptorByTag(tag).deserialize(this, codedIn)
        : constant;
  }

  /**
   * Returns a {@link MemoizingCodec} appropriate for deserializing the next object on the wire.
   * Only for use by {@link Memoizer.Deserializer}.
   */
  @SuppressWarnings("unchecked")
  public MemoizingCodec<?> getMemoizingCodecRecordedInInput(CodedInputStream codedIn)
      throws IOException, SerializationException {
    int tag = codedIn.readSInt32();
    if (tag == 0) {
      return new InitializedMemoizingCodec<>(null);
    }
    Object constant = registry.maybeGetConstantByTag(tag);
    if (constant != null) {
      return new InitializedMemoizingCodec<>(constant);
    }
    MemoizingCodec<?> codec = registry.maybeGetMemoizingCodecByTag(tag);
    if (codec != null) {
      return codec;
    }
    return new MemoizingAfterObjectCodecAdapter<>(registry.getCodecDescriptorByTag(tag).getCodec());
  }

  private static class InitializedMemoizingCodec<T> implements MemoizingCodec<T> {
    private final T obj;
    private boolean deserialized = false;

    private InitializedMemoizingCodec(T obj) {
      this.obj = obj;
    }

    @Override
    public Strategy getStrategy() {
      return Strategy.DO_NOT_MEMOIZE;
    }

    @Override
    public void serializePayload(
        SerializationContext context, T obj, CodedOutputStream codedOut, Serializer serializer) {
      throw new UnsupportedOperationException("Unexpected serialize: " + obj);
    }

    @Override
    public T deserializePayload(
        DeserializationContext context,
        @Nullable T initial,
        CodedInputStream codedIn,
        Deserializer deserializer) {
      Preconditions.checkState(!deserialized, "Already deserialized: %s", obj);
      deserialized = true;
      return obj;
    }

    @SuppressWarnings("unchecked")
    @Override
    public Class<T> getEncodedClass() {
      return (Class<T>) obj.getClass();
    }
  }
  @SuppressWarnings("unchecked")
  public <T> T getDependency(Class<T> type) {
    Preconditions.checkNotNull(type);
    return (T) dependencies.get(type);
  }
}
