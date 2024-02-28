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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.HashSet;
import java.util.Set;
import javax.annotation.Nullable;

/** A stateful serializer that emits backreferences instead of reserializing duplicate objects. */
abstract class MemoizingSerializationContext extends SerializationContext {
  private final Memoizer.Serializer serializer = new Memoizer.Serializer();
  private final Set<Class<?>> explicitlyAllowedClasses = new HashSet<>();

  @VisibleForTesting // private
  static MemoizingSerializationContext createForTesting(
      ObjectCodecRegistry codecRegistry, ImmutableClassToInstanceMap<Object> dependencies) {
    return new MemoizingSerializationContextImpl(codecRegistry, dependencies);
  }

  MemoizingSerializationContext(
      ObjectCodecRegistry codecRegistry, ImmutableClassToInstanceMap<Object> dependencies) {
    super(codecRegistry, dependencies);
  }

  static byte[] serializeToBytes(
      ObjectCodecRegistry codecRegistry,
      ImmutableClassToInstanceMap<Object> dependencies,
      @Nullable Object subject,
      int outputCapacity,
      int bufferCapacity)
      throws SerializationException {
    ByteArrayOutputStream bytesOut = new ByteArrayOutputStream(outputCapacity);
    serializeToStream(codecRegistry, dependencies, subject, bytesOut, bufferCapacity);
    return bytesOut.toByteArray();
  }

  static ByteString serializeToByteString(
      ObjectCodecRegistry codecRegistry,
      ImmutableClassToInstanceMap<Object> dependencies,
      @Nullable Object subject,
      int outputCapacity,
      int bufferCapacity)
      throws SerializationException {
    ByteString.Output bytesOut = ByteString.newOutput(outputCapacity);
    serializeToStream(codecRegistry, dependencies, subject, bytesOut, bufferCapacity);
    return bytesOut.toByteString();
  }

  @Override
  public final void addExplicitlyAllowedClass(Class<?> allowedClass) {
    explicitlyAllowedClasses.add(allowedClass);
  }

  @Override
  public final <T> void checkClassExplicitlyAllowed(Class<T> allowedClass, T objectForDebugging)
      throws SerializationException {
    if (!explicitlyAllowedClasses.contains(allowedClass)) {
      throw new SerializationException(
          allowedClass
              + " not explicitly allowed (allowed classes were: "
              + explicitlyAllowedClasses
              + ") and object is "
              + objectForDebugging);
    }
  }

  @Override
  final void serializeWithCodec(ObjectCodec<Object> codec, Object obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    serializer.serialize(this, obj, codec, codedOut);
  }

  @Override
  final boolean writeBackReferenceIfMemoized(Object obj, CodedOutputStream codedOut)
      throws IOException {
    int memoizedIndex = serializer.getMemoizedIndex(obj);
    if (memoizedIndex == -1) {
      return false;
    }
    // Subtracts 1 so it will be negative and not collide with null.
    codedOut.writeSInt32NoTag(-memoizedIndex - 1);
    return true;
  }

  @Override
  public final boolean isMemoizing() {
    return true;
  }

  private static void serializeToStream(
      ObjectCodecRegistry codecRegistry,
      ImmutableClassToInstanceMap<Object> dependencies,
      @Nullable Object subject,
      OutputStream output,
      int bufferCapacity)
      throws SerializationException {
    CodedOutputStream codedOut = CodedOutputStream.newInstance(output, bufferCapacity);
    try {
      new MemoizingSerializationContextImpl(codecRegistry, dependencies)
          .serialize(subject, codedOut);
      codedOut.flush();
    } catch (IOException e) {
      throw new SerializationException("Failed to serialize " + subject, e);
    }
  }

  /**
   * This mainly exists to restrict use of {@link MemoizingSerializationContext}'s constructor.
   *
   * <p>It's also slightly cleaner for {@link SharedValueSerializationContext} to not inherit the
   * implementation of {@link #getFreshContext}.
   */
  private static final class MemoizingSerializationContextImpl
      extends MemoizingSerializationContext {
    private MemoizingSerializationContextImpl(
        ObjectCodecRegistry codecRegistry, ImmutableClassToInstanceMap<Object> dependencies) {
      super(codecRegistry, dependencies);
    }

    @Override
    public MemoizingSerializationContext getFreshContext() {
      return new MemoizingSerializationContextImpl(getCodecRegistry(), getDependencies());
    }
  }
}
