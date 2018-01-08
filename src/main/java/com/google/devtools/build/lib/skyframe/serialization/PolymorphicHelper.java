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

import com.google.devtools.build.lib.skyframe.serialization.strings.StringCodecs;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.Optional;
import javax.annotation.Nullable;

/** Helper methods for polymorphic codecs using reflection. */
public class PolymorphicHelper {

  private PolymorphicHelper() {}

  /**
   * Serializes a polymorphic type.
   *
   * @param dependency if null, it means that the parent, polymorphic type, provides no dependency.
   *     It is valid for dependency to be non-null, with an enclosed null value. This means that
   *     dependency itself is null (as opposed to non-existent).
   */
  @SuppressWarnings("unchecked")
  public static void serialize(
      Object input, CodedOutputStream codedOut, @Nullable Optional<?> dependency)
      throws IOException, SerializationException {
    if (input != null) {
      Class<?> clazz = input.getClass();
      try {
        Object codec = getCodec(clazz);
        codedOut.writeBoolNoTag(true);
        StringCodecs.asciiOptimized().serialize(clazz.getName(), codedOut);
        if (codec instanceof ObjectCodec) {
          ((ObjectCodec) codec).serialize(input, codedOut);
        } else if (codec instanceof InjectingObjectCodec) {
          if (dependency == null) {
            throw new SerializationException(
                clazz.getCanonicalName() + " serialize parent class lacks required dependency.");
          }
          ((InjectingObjectCodec) codec).serialize(dependency.orElse(null), input, codedOut);
        } else {
          throw new SerializationException(
              clazz.getCanonicalName()
                  + ".CODEC has unexpected type "
                  + codec.getClass().getCanonicalName());
        }
      } catch (ReflectiveOperationException e) {
        throw new SerializationException(input.getClass().getName(), e);
      }
    } else {
      codedOut.writeBoolNoTag(false);
    }
  }

  /**
   * Deserializes a polymorphic type.
   *
   * @param dependency if null, it means that the parent, polymorphic type, provides no dependency.
   *     It is valid for dependency to be non-null, with an enclosed null value. This means the
   *     dependency itself is null (as opposed to non-existent).
   */
  @SuppressWarnings("unchecked")
  public static Object deserialize(CodedInputStream codedIn, @Nullable Optional<?> dependency)
      throws IOException, SerializationException {
    Object deserialized = null;
    if (codedIn.readBool()) {
      String className = StringCodecs.asciiOptimized().deserialize(codedIn);
      try {
        Object codec = getCodec(Class.forName(className));
        if (codec instanceof ObjectCodec) {
          return ((ObjectCodec) codec).deserialize(codedIn);
        } else if (codec instanceof InjectingObjectCodec) {
          if (dependency == null) {
            throw new SerializationException(
                className + " deserialize parent class lacks required dependency.");
          }
          return ((InjectingObjectCodec) codec).deserialize(dependency.orElse(null), codedIn);
        } else {
          throw new SerializationException(
              className + ".CODEC has unexpected type " + codec.getClass().getCanonicalName());
        }
      } catch (ReflectiveOperationException e) {
        throw new SerializationException(className, e);
      }
    }
    return deserialized;
  }

  /** Returns the static CODEC instance for {@code clazz}. */
  private static Object getCodec(Class<?> clazz)
      throws NoSuchFieldException, IllegalAccessException {
    Field codecField = clazz.getDeclaredField("CODEC");
    codecField.setAccessible(true);
    return codecField.get(null);
  }
}
