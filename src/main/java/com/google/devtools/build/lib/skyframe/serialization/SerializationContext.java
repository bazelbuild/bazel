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
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import javax.annotation.Nullable;

/** Stateful class for providing additional context to a single serialization "session". */
// TODO(bazel-team): This class is just a shell, fill in.
public class SerializationContext {

  private final ObjectCodecRegistry registry;
  private final ImmutableMap<Class<?>, Object> dependencies;

  public SerializationContext(
      ObjectCodecRegistry registry, ImmutableMap<Class<?>, Object> dependencies) {
    this.registry = registry;
    this.dependencies = dependencies;
  }

  public SerializationContext(ImmutableMap<Class<?>, Object> dependencies) {
    this(AutoRegistry.get(), dependencies);
  }

  @Nullable
  private ObjectCodecRegistry.CodecDescriptor recordAndGetDescriptorIfNotConstantOrNull(
      @Nullable Object object, CodedOutputStream codedOut) throws IOException, NoCodecException {
    if (object == null) {
      codedOut.writeSInt32NoTag(0);
      return null;
    }
    Integer tag = registry.maybeGetTagForConstant(object);
    if (tag != null) {
      codedOut.writeSInt32NoTag(tag);
      return null;
    }
    ObjectCodecRegistry.CodecDescriptor descriptor = registry.getCodecDescriptor(object.getClass());
    codedOut.writeSInt32NoTag(descriptor.getTag());
    return descriptor;
  }

  // TODO(shahan): consider making codedOut a member of this class.
  public void serialize(Object object, CodedOutputStream codedOut)
      throws IOException, SerializationException {
    ObjectCodecRegistry.CodecDescriptor descriptor =
        recordAndGetDescriptorIfNotConstantOrNull(object, codedOut);
    if (descriptor != null) {
      descriptor.serialize(this, object, codedOut);
    }
  }

  /**
   * Returns the {@link ObjectCodec} to use when serializing {@code object}. The codec's tag is
   * written to {@code codedOut} so that {@link DeserializationContext#getCodecRecordedInInput} can
   * return the correct codec as well. Only to be used by {@code MemoizingCodec}.
   *
   * <p>If the return value is null, this indicates that the value was serialized directly as null
   * or a constant. The caller needs do nothing further to serialize {@code object} in this case.
   */
  @SuppressWarnings("unchecked") // cast to ObjectCodec<? super T>.
  public <T> ObjectCodec<? super T> recordAndMaybeReturnCodec(T object, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    ObjectCodecRegistry.CodecDescriptor descriptor =
        recordAndGetDescriptorIfNotConstantOrNull(object, codedOut);
    if (descriptor == null) {
      return null;
    }
    return (ObjectCodec<? super T>) descriptor.getCodec();
  }

  @SuppressWarnings("unchecked")
  public <T> T getDependency(Class<T> type) {
    Preconditions.checkNotNull(type);
    return (T) dependencies.get(type);
  }
}
