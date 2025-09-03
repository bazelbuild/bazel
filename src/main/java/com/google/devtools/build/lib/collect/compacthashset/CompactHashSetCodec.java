// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.collect.compacthashset;

import com.google.devtools.build.lib.skyframe.serialization.AsyncDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/** Codec implementation for {@link CompactHashSet}. */
@SuppressWarnings("rawtypes")
final class CompactHashSetCodec extends DeferredObjectCodec<CompactHashSet> {
  @Override
  public Class<CompactHashSet> getEncodedClass() {
    return CompactHashSet.class;
  }

  @Override
  public void serialize(
      SerializationContext context, CompactHashSet obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    codedOut.writeInt32NoTag(obj.size());
    for (Object elt : obj) {
      context.serialize(elt, codedOut);
    }
  }

  @Override
  public DeferredValue<CompactHashSet> deserializeDeferred(
      AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int size = codedIn.readInt32();
    var builder = new Builder(size);
    for (int i = 0; i < size; i++) {
      context.deserializeArrayElement(codedIn, builder.elements, i);
    }
    return builder;
  }

  private static final class Builder implements DeferredValue<CompactHashSet> {
    private final Object[] elements;

    private Builder(int size) {
      this.elements = new Object[size];
    }

    @Override
    public CompactHashSet call() {
      return CompactHashSet.create(elements);
    }
  }
}
