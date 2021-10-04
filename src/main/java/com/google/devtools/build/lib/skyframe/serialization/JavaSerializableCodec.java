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

import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import com.google.protobuf.MessageLite;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.NotSerializableException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.ByteBuffer;

/** Naive ObjectCodec using Java native Serialization. Not performant, but a good fallback */
class JavaSerializableCodec implements ObjectCodec<Object> {

  private static boolean shouldNotBeJavaSerialized(Class<?> clazz) {
    return MessageLite.class.isAssignableFrom(clazz);
  }

  @Override
  public Class<Object> getEncodedClass() {
    return Object.class;
  }

  @Override
  public void serialize(SerializationContext context, Object obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    if (shouldNotBeJavaSerialized(obj.getClass())) {
      throw new SerializationException(
          "Java serialization is not permitted for class " + obj.getClass());
    }
    ByteString.Output out = ByteString.newOutput();
    ObjectOutputStream objOut = new ObjectOutputStream(out);
    try {
      objOut.writeObject(obj);
    } catch (NotSerializableException e) {
      Class<?> clazz = obj.getClass();
      Class<?> parentClass = null;
      if (clazz.isAnonymousClass() || clazz.isSynthetic()) {
        parentClass = clazz.getSuperclass();
      }
      throw new SerializationException.NoCodecException(
          "Object "
              + obj
              + " of type "
              + obj.getClass()
              + (parentClass == null ? "" : " (parent " + parentClass + ")")
              + " not serializable",
          e);
    } catch (NotSerializableRuntimeException e) {
      // Values that inherit from Serializable but actually aren't serializable.
      throw new SerializationException.NoCodecException(
          "Object " + obj + " of type " + obj.getClass() + " not serializable", e);
    }
    codedOut.writeBytesNoTag(out.toByteString());
  }

  @Override
  public Object deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    try {
      // Get the ByteBuffer as it is potentially a view of the underlying bytes (not a copy), which
      // is more efficient.
      ByteBuffer buffer = codedIn.readByteBuffer();
      ObjectInputStream objIn =
          new ObjectInputStream(
              new ByteArrayInputStream(buffer.array(), buffer.arrayOffset(), buffer.remaining()));
      return objIn.readObject();
    } catch (ClassNotFoundException e) {
      throw new SerializationException("Java deserialization failed", e);
    }
  }

  /** Disables auto-registration. */
  @SuppressWarnings("unused") // Used reflectively.
  private static class JavaSerializableCodecRegisterer
      implements CodecRegisterer<JavaSerializableCodec> {}
}
