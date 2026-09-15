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
package com.google.devtools.build.lib.collect;

import static com.google.devtools.build.lib.skyframe.serialization.ArrayProcessor.deserializeObjectArray;

import com.google.common.collect.FluentIterable;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.skyframe.serialization.AsyncDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.AsyncObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.Collection;
import java.util.Iterator;

@SuppressWarnings({"rawtypes", "unchecked"})
class IterableCodecs {
  /**
   * Codec for {@link FluentIterable}.
   *
   * <p>{@link FluentIterable} isn't directly used, but instances are created through {@link
   * com.google.common.collect.Iterables#concat}.
   */
  static class FluentIterableCodec extends AsyncObjectCodec<FluentIterable> {
    @Override
    public Class<FluentIterable> getEncodedClass() {
      return FluentIterable.class;
    }

    @Override
    public void serialize(
        SerializationContext context, FluentIterable obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      IterableCodecs.serialize(context, obj, codedOut);
    }

    @Override
    public FluentIterable deserializeAsync(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      int count = codedIn.readInt32();
      if (count == 0) {
        FluentIterable empty = FluentIterable.of();
        context.registerInitialValue(empty);
        return empty;
      }

      Object[] elements = new Object[count];
      DeserializedFluentIterable value = new DeserializedFluentIterable(elements);
      context.registerInitialValue(value);

      deserializeObjectArray(context, codedIn, elements, count);

      return value;
    }
  }

  /** This imitates the anonymous implementation in {@link FluentIterable#from}. */
  private static final class DeserializedFluentIterable extends FluentIterable {
    private final Object[] elements;

    private DeserializedFluentIterable(Object[] elements) {
      this.elements = elements;
    }

    @Override
    public Iterator iterator() {
      return Iterators.forArray(elements);
    }
  }

  private static void serialize(
      SerializationContext context, Iterable obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    Collection elements = obj instanceof Collection ? (Collection) obj : Lists.newArrayList(obj);
    codedOut.writeInt32NoTag(elements.size());
    for (Object elt : elements) {
      context.serialize(elt, codedOut);
    }
  }
}
