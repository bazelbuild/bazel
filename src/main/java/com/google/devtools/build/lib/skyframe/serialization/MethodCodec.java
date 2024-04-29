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

import static com.google.devtools.build.lib.skyframe.serialization.ClassCodec.classCodec;
import static com.google.devtools.build.lib.skyframe.serialization.strings.UnsafeStringCodec.stringCodec;

import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.lang.reflect.Method;
import java.util.Arrays;

/** {@link ObjectCodec} for {@link Method}. */
class MethodCodec extends LeafObjectCodec<Method> {
  @Override
  public Class<Method> getEncodedClass() {
    return Method.class;
  }

  @Override
  public void serialize(LeafSerializationContext context, Method obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    context.serializeLeaf(obj.getDeclaringClass(), classCodec(), codedOut);
    context.serializeLeaf(obj.getName(), stringCodec(), codedOut);
    Class<?>[] parameterTypes = obj.getParameterTypes();
    codedOut.writeInt32NoTag(parameterTypes.length);
    for (Class<?> parameter : parameterTypes) {
      context.serializeLeaf(parameter, classCodec(), codedOut);
    }
  }

  @Override
  public Method deserialize(LeafDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    Class<?> clazz = context.deserializeLeaf(codedIn, classCodec());
    String name = context.deserializeLeaf(codedIn, stringCodec());

    Class<?>[] parameters = new Class<?>[codedIn.readInt32()];
    for (int i = 0; i < parameters.length; i++) {
      parameters[i] = context.deserializeLeaf(codedIn, classCodec());
    }
    try {
      return clazz.getDeclaredMethod(name, parameters);
    } catch (NoSuchMethodException e) {
      throw new SerializationException(
          "Couldn't get method "
              + name
              + " in "
              + clazz
              + " with parameters "
              + Arrays.toString(parameters),
          e);
    }
  }
}
