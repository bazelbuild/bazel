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

import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/** {@link ObjectCodec} for {@link Object[]}. */
class ArrayCodec implements ObjectCodec<Object[]> {
  @Override
  public Class<Object[]> getEncodedClass() {
    return Object[].class;
  }

  @Override
  public void serialize(SerializationContext context, Object[] obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    codedOut.writeInt32NoTag(obj.length);
    try {
      for (Object item : obj) {
        context.serialize(item, codedOut);
      }
    } catch (StackOverflowError e) {
      // TODO(janakr): figure out if we need to handle this better and handle it better if so.
      throw new SerializationException("StackOverflow serializing array", e);
    }
  }

  @Override
  public Object[] deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    Object[] result = new Object[codedIn.readInt32()];
    try {
      for (int i = 0; i < result.length; i++) {
        result[i] = context.deserialize(codedIn);
      }
    } catch (StackOverflowError e) {
      // TODO(janakr): figure out if we need to handle this better and handle it better if so.
      throw new SerializationException("StackOverflow deserializing array", e);
    }
    return result;
  }
}
