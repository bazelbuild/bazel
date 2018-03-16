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
import java.util.ArrayList;

@SuppressWarnings("rawtypes")
class ArrayListCodec implements ObjectCodec<ArrayList> {

  @Override
  public Class<ArrayList> getEncodedClass() {
    return ArrayList.class;
  }

  @Override
  public void serialize(SerializationContext context, ArrayList list, CodedOutputStream codedOut)
      throws IOException, SerializationException {
    codedOut.writeInt32NoTag(list.size());
    for (Object obj : list) {
      context.serialize(obj, codedOut);
    }
  }

  @Override
  public ArrayList deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int length = codedIn.readInt32();
    if (length < 0) {
      throw new SerializationException("Expected non-negative length: " + length);
    }

    ArrayList<Object> list = new ArrayList<>(length);
    for (int i = 0; i < length; ++i) {
      list.add(context.deserialize(codedIn));
    }
    return list;
  }
}
