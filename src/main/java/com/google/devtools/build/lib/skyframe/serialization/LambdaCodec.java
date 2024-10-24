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

import static com.google.devtools.build.lib.unsafe.UnsafeProvider.getFieldOffset;

import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.io.Serializable;
import java.lang.invoke.SerializedLambda;
import java.lang.reflect.Method;

/**
 * A codec for Java 8 serializable lambdas. Lambdas that are tagged as {@link Serializable} have a
 * generated method, {@code writeReplace}, that converts them into a {@link SerializedLambda}, which
 * can then be serialized like a normal object. On deserialization, we call {@link
 * SerializedLambda#readResolve}, which converts the object back into a lambda.
 *
 * <p>Since lambdas do not share a common base class, choosing this codec for serializing them must
 * be special-cased in {@link ObjectCodecRegistry}. We must also make a somewhat arbitrary choice
 * around the generic parameter. Since all of our lambdas are {@link Serializable}, we use that.
 * Because {@link Serializable} is an interface, not a class, this codec will never be chosen for
 * any object without special-casing.
 */
class LambdaCodec extends DeferredObjectCodec<Serializable> {
  private static final Method READ_RESOLVE_METHOD;
  private static final long SERIALIZED_LAMBDA_OFFSET;

  static {
    try {
      READ_RESOLVE_METHOD = SerializedLambda.class.getDeclaredMethod("readResolve");
      SERIALIZED_LAMBDA_OFFSET = getFieldOffset(LambdaSupplier.class, "serializedLambda");
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
    READ_RESOLVE_METHOD.setAccessible(true);
  }

  static boolean isProbablyLambda(Class<?> type) {
    return type.isSynthetic() && !type.isLocalClass() && !type.isAnonymousClass();
  }

  @Override
  public Class<? extends Serializable> getEncodedClass() {
    return Serializable.class;
  }

  @Override
  public void serialize(SerializationContext context, Serializable obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    Class<?> objClass = obj.getClass();
    if (!isProbablyLambda(objClass)) {
      throw new SerializationException(obj + " is not a lambda: " + objClass);
    }
    Method writeReplaceMethod;
    try {
      // TODO(janakr): We could cache these methods if retrieval shows up as a hotspot.
      writeReplaceMethod = objClass.getDeclaredMethod("writeReplace");
    } catch (NoSuchMethodException e) {
      throw new SerializationException(
          "No writeReplace method for " + obj + " with " + objClass, e);
    }
    writeReplaceMethod.setAccessible(true);
    SerializedLambda serializedLambda;
    try {
      serializedLambda = (SerializedLambda) writeReplaceMethod.invoke(obj);
    } catch (ReflectiveOperationException e) {
      throw new SerializationException(
          "Exception invoking writeReplace for " + obj + " with " + objClass, e);
    }
    context.serialize(serializedLambda, codedOut);
  }

  @Override
  public DeferredValue<Serializable> deserializeDeferred(
      AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    LambdaSupplier supplier = new LambdaSupplier();
    context.deserialize(codedIn, supplier, SERIALIZED_LAMBDA_OFFSET);
    return supplier;
  }

  private static class LambdaSupplier implements DeferredValue<Serializable> {
    private SerializedLambda serializedLambda;

    @Override
    public Serializable call() {
      try {
        return (Serializable) READ_RESOLVE_METHOD.invoke(serializedLambda);
      } catch (ReflectiveOperationException e) {
        throw new IllegalStateException("Error read-resolving " + serializedLambda, e);
      }
    }
  }
}
