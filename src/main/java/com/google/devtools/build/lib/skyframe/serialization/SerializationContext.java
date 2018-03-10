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
import com.google.devtools.build.lib.skyframe.serialization.Memoizer.MemoizingAfterObjectCodecAdapter;
import com.google.devtools.build.lib.skyframe.serialization.Memoizer.MemoizingCodec;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecRegistry.MemoizingCodecDescriptor;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException.NoCodecException;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import javax.annotation.Nullable;

/** Stateful class for providing additional context to a single serialization "session". */
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

  private boolean writeNullOrConstant(@Nullable Object object, CodedOutputStream codedOut)
      throws IOException {
    if (object == null) {
      codedOut.writeSInt32NoTag(0);
      return true;
    }
    Integer tag = registry.maybeGetTagForConstant(object);
    if (tag != null) {
      codedOut.writeSInt32NoTag(tag);
      return true;
    }
    return false;
  }

  @Nullable
  private ObjectCodecRegistry.CodecDescriptor recordAndGetDescriptorIfNotConstantOrNull(
      @Nullable Object object, CodedOutputStream codedOut) throws IOException, NoCodecException {
    if (writeNullOrConstant(object, codedOut)) {
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

  @Nullable
  public <T> MemoizingCodec<? super T> recordAndMaybeGetMemoizingCodec(
      T object, CodedOutputStream codedOut) throws IOException, NoCodecException {
    if (writeNullOrConstant(object, codedOut)) {
      return null;
    }
    @SuppressWarnings("unchecked")
    Class<T> clazz = (Class<T>) object.getClass();
    MemoizingCodecDescriptor<? super T> memoizingCodecDescriptor =
        registry.getMemoizingCodecDescriptor(clazz);
    if (memoizingCodecDescriptor != null) {
      codedOut.writeSInt32NoTag(memoizingCodecDescriptor.getTag());
      return memoizingCodecDescriptor.getMemoizingCodec();
    }
    ObjectCodecRegistry.CodecDescriptor codecDescriptor = registry.getCodecDescriptor(clazz);
    codedOut.writeSInt32NoTag(codecDescriptor.getTag());
    @SuppressWarnings("unchecked")
    ObjectCodec<? super T> objectCodec = (ObjectCodec<? super T>) codecDescriptor.getCodec();
    return new MemoizingAfterObjectCodecAdapter<>(objectCodec);
  }

  @SuppressWarnings("unchecked")
  public <T> T getDependency(Class<T> type) {
    Preconditions.checkNotNull(type);
    return (T) dependencies.get(type);
  }
}
