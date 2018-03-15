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

import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.ByteBuffer;

@SuppressWarnings("rawtypes")
class EnumRuntimeCodec implements ObjectCodec<Enum> {

  @Override
  public Class<Enum> getEncodedClass() {
    return Enum.class;
  }

  @Override
  public void serialize(SerializationContext unusedContext, Enum value, CodedOutputStream codedOut)
      throws IOException, SerializationException {
    // We're using Java serialization below because Enums are serializable by default and a
    // hand-rolled version is going to be very similar.
    ByteString.Output out = ByteString.newOutput();
    ObjectOutputStream objOut = new ObjectOutputStream(out);
    objOut.writeObject(value);
    codedOut.writeBytesNoTag(out.toByteString());
  }

  @Override
  public Enum deserialize(DeserializationContext unusedContext, CodedInputStream codedIn)
      throws SerializationException, IOException {
    ByteBuffer buffer = codedIn.readByteBuffer();
    ObjectInputStream objIn =
        new ObjectInputStream(
            new ByteArrayInputStream(buffer.array(), buffer.arrayOffset(), buffer.remaining()));
    try {
      return (Enum) objIn.readObject();
    } catch (ClassNotFoundException e) {
      throw new SerializationException("Couldn't find class for Enum deserialization?", e);
    }
  }
}
