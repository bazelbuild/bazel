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

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.KryoException;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

/** Converts an {@link ObjectCodec} into a Kryo {@link Serializer}. */
public class SerializerAdapter<T> extends Serializer<T> {
  private final ObjectCodec<T> codec;

  public SerializerAdapter(ObjectCodec<T> codec) {
    this.codec = codec;
  }

  @Override
  public void write(Kryo kryo, Output output, T object) {
    try {
      ByteArrayOutputStream byteOutput = new ByteArrayOutputStream();
      CodedOutputStream codedOut = CodedOutputStream.newInstance(byteOutput);
      // TODO(shahan): Determine if there's any context we can/should pass along from kryo.
      codec.serialize(SerializationContext.UNTHREADED_PLEASE_FIX, object, codedOut);
      codedOut.flush();
      byte[] byteData = byteOutput.toByteArray();
      output.writeInt(byteData.length, true);
      output.writeBytes(byteData);
    } catch (SerializationException | IOException e) {
      throw new KryoException(e);
    }
  }

  @Override
  public T read(Kryo kryo, Input input, Class<T> unusedClass) {
    try {
      byte[] byteData = input.readBytes(input.readInt(true));
      // TODO(shahan): Determine if there's any context we can/should pass along from kryo.
      return codec.deserialize(
          DeserializationContext.UNTHREADED_PLEASE_FIX, CodedInputStream.newInstance(byteData));
    } catch (SerializationException | IOException e) {
      throw new KryoException(e);
    }
  }
}
