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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.devtools.build.lib.skyframe.serialization.ArrayProcessor.deserializeObjectArray;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.List;

/**
 * Codec that serializes the hidden, {@link Lists.TransformingRandomAccessList} type.
 *
 * <p>Note that {@link List} is an interface, so codec resolution will never match it. Since {@link
 * Lists.TransformingRandomAccessList} is hidden, it's safe to deserialize as {@link ImmutableList},
 * because it must be referenced as a {@link List} or something more general.
 */
@SuppressWarnings("rawtypes")
final class TransformingRandomAccessListCodec extends DeferredObjectCodec<List> {
  private static final Class<? extends List> TRANSFORMING_RANDOM_ACCESS_LIST_TYPE =
      Lists.transform(ImmutableList.<Integer>of(1, 2, 3), x -> x).getClass();

  @Override
  public Class<? extends List> getEncodedClass() {
    return TRANSFORMING_RANDOM_ACCESS_LIST_TYPE;
  }

  @Override
  public void serialize(SerializationContext context, List object, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    codedOut.writeInt32NoTag(object.size());
    for (Object obj : object) {
      context.serialize(obj, codedOut);
    }
  }

  @Override
  public DeferredValue<List> deserializeDeferred(
      AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int size = codedIn.readInt32();

    ElementBuffer buffer = new ElementBuffer(size);
    deserializeObjectArray(context, codedIn, buffer.elements, size);
    return buffer;
  }

  private static class ElementBuffer implements DeferredValue<List> {
    private final Object[] elements;

    private ElementBuffer(int size) {
      this.elements = new Object[size];
    }

    @Override
    public ImmutableList call() {
      return ImmutableList.builderWithExpectedSize(elements.length).add(elements).build();
    }
  }

  private TransformingRandomAccessListCodec() {}
}
