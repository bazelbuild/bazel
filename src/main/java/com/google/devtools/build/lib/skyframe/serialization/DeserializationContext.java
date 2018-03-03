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
import com.google.devtools.build.lib.skyframe.serialization.SerializationException.NoCodecException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

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
   * Returns an {@link ObjectCodec} appropriate for deserializing the next object on the wire. Only
   * to be used by {@code MemoizingCodec}.
   */
  @SuppressWarnings("unchecked")
  public <T> ObjectCodec<T> getCodecRecordedInInput(CodedInputStream codedIn)
      throws IOException, NoCodecException {
    int tag = codedIn.readSInt32();
    if (tag == 0) {
      return null;
    }
    T constant = (T) registry.maybeGetConstantByTag(tag);
    return constant == null
        ? (ObjectCodec<T>) registry.getCodecDescriptorByTag(tag).getCodec()
        : new InitializedObjectCodec<>(constant);
  }

  private static class InitializedObjectCodec<T> implements ObjectCodec<T> {
    private final T obj;
    private boolean deserialized = false;

    private InitializedObjectCodec(T obj) {
      this.obj = obj;
    }

    @Override
    public void serialize(SerializationContext context, T obj, CodedOutputStream codedOut) {
      throw new UnsupportedOperationException("Unexpected serialize: " + obj);
    }

    @Override
    public T deserialize(DeserializationContext context, CodedInputStream codedIn) {
      Preconditions.checkState(!deserialized, "Deserialize called twice: %s", obj);
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
