// Copyright 2023 The Bazel Authors. All rights reserved.
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
import java.util.concurrent.atomic.AtomicReferenceArray;

final class AtomicReferenceArrayCodec<V> implements ObjectCodec<AtomicReferenceArray<V>> {
  @SuppressWarnings("unchecked")
  @Override
  public Class<AtomicReferenceArray<V>> getEncodedClass() {
    return (Class<AtomicReferenceArray<V>>) (Class<?>) AtomicReferenceArray.class;
  }

  @Override
  public void serialize(
      SerializationContext context, AtomicReferenceArray<V> obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    int length = obj.length();
    codedOut.writeInt32NoTag(length);
    for (int i = 0; i < length; i++) {
      context.serialize(obj.getPlain(i), codedOut);
    }
  }

  @Override
  public AtomicReferenceArray<V> deserialize(
      DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int length = codedIn.readInt32();
    AtomicReferenceArray<V> result = new AtomicReferenceArray<V>(length);
    for (int i = 0; i < length; i++) {
      result.setPlain(i, context.deserialize(codedIn));
    }
    return result;
  }
}
