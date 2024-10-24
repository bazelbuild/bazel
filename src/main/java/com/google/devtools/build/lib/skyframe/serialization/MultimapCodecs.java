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

import static com.google.devtools.build.lib.skyframe.serialization.ArrayProcessor.deserializeObjectArray;

import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec.DeferredValue;
import com.google.errorprone.annotations.Keep;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.Map;

/**
 * Codecs for {@link Multimap}. Handles {@link ImmutableListMultimap}, {@link ImmutableSetMultimap}
 * and {@link LinkedHashMultimap}.
 */
@SuppressWarnings({"unchecked", "rawtypes", "NonApiType"})
public final class MultimapCodecs {
  @Keep // used reflectively
  private static class ImmutableListMultimapCodec
      extends DeferredObjectCodec<ImmutableListMultimap> {
    @Override
    public Class<ImmutableListMultimap> getEncodedClass() {
      return ImmutableListMultimap.class;
    }

    @Override
    public void serialize(
        SerializationContext context, ImmutableListMultimap obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      serializeMultimap(context, obj, codedOut);
    }

    @Override
    public DeferredValue<ImmutableListMultimap> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      int size = codedIn.readInt32();
      if (size == 0) {
        return ImmutableListMultimap::of;
      }

      ImmutableListMultimapBuffer buffer = new ImmutableListMultimapBuffer(size);
      for (int i = 0; i < size; i++) {
        context.deserializeArrayElement(codedIn, buffer.keys, i);
        int valuesCount = codedIn.readInt32();
        Object[] values = new Object[valuesCount];
        buffer.values[i] = values;
        // The builder merely collects these references in an ArrayList, so (unlike the set-type
        // multimaps) these values do not need to be fully deserialized.
        deserializeObjectArray(context, codedIn, buffer.values[i], valuesCount);
      }
      return buffer;
    }
  }

  @Keep // used reflectively
  private static class ImmutableSetMultimapCodec extends DeferredObjectCodec<ImmutableSetMultimap> {
    @Override
    public Class<ImmutableSetMultimap> getEncodedClass() {
      return ImmutableSetMultimap.class;
    }

    @Override
    public void serialize(
        SerializationContext context, ImmutableSetMultimap obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      serializeMultimap(context, obj, codedOut);
    }

    @Override
    public DeferredValue<ImmutableSetMultimap> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      int size = codedIn.readInt32();
      if (size == 0) {
        return ImmutableSetMultimap::of;
      }

      ImmutableSetMultimapBuffer buffer = new ImmutableSetMultimapBuffer(size);
      deserializeSetMultimap(context, codedIn, buffer);
      return buffer;
    }
  }

  @Keep // used reflectively
  private static class LinkedHashMultimapCodec extends DeferredObjectCodec<LinkedHashMultimap> {
    @Override
    public Class<LinkedHashMultimap> getEncodedClass() {
      return LinkedHashMultimap.class;
    }

    @Override
    public void serialize(
        SerializationContext context, LinkedHashMultimap obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      serializeMultimap(context, obj, codedOut);
    }

    @Override
    public DeferredValue<LinkedHashMultimap> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      int size = codedIn.readInt32();
      if (size == 0) {
        return LinkedHashMultimap::create;
      }

      LinkedHashMultimapBuffer buffer = new LinkedHashMultimapBuffer(size);
      deserializeSetMultimap(context, codedIn, buffer);
      return buffer;
    }
  }

  private static void serializeMultimap(
      SerializationContext context, Multimap obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    Map map = obj.asMap();
    codedOut.writeInt32NoTag(map.size());
    for (Object next : map.entrySet()) {
      Map.Entry entry = (Map.Entry) next;

      context.serialize(entry.getKey(), codedOut);

      Collection values = (Collection) entry.getValue();
      codedOut.writeInt32NoTag(values.size());
      for (Object value : values) {
        context.serialize(value, codedOut);
      }
    }
  }

  /** Takes care to fully deserialize all keys and values as they will be used in sets. */
  private static void deserializeSetMultimap(
      AsyncDeserializationContext context, CodedInputStream codedIn, MultimapBuffer buffer)
      throws SerializationException, IOException {
    for (int i = 0; i < buffer.size(); i++) {
      context.deserializeArrayElement(codedIn, buffer.keys, i);

      int valuesCount = codedIn.readInt32();
      Object[] values = new Object[valuesCount];
      buffer.values[i] = values;
      // The builder uses a set to collect the values so they must be complete.
      deserializeObjectArray(context, codedIn, values, valuesCount);
    }
  }

  private static class MultimapBuffer {
    final Object[] keys;
    final Object[][] values;

    private MultimapBuffer(int size) {
      this.keys = new Object[size];
      this.values = new Object[size][];
    }

    int size() {
      return keys.length;
    }
  }

  private static class ImmutableListMultimapBuffer extends MultimapBuffer
      implements DeferredValue<ImmutableListMultimap> {
    private ImmutableListMultimapBuffer(int size) {
      super(size);
    }

    @Override
    public ImmutableListMultimap call() {
      ImmutableListMultimap.Builder builder = ImmutableListMultimap.builder();
      for (int i = 0; i < size(); i++) {
        builder.putAll(keys[i], values[i]);
      }
      return builder.build();
    }
  }

  private static class ImmutableSetMultimapBuffer extends MultimapBuffer
      implements DeferredValue<ImmutableSetMultimap> {
    private ImmutableSetMultimapBuffer(int size) {
      super(size);
    }

    @Override
    public ImmutableSetMultimap call() {
      ImmutableSetMultimap.Builder builder = ImmutableSetMultimap.builder();
      for (int i = 0; i < size(); i++) {
        builder.putAll(keys[i], values[i]);
      }
      return builder.build();
    }
  }

  private static class LinkedHashMultimapBuffer extends MultimapBuffer
      implements DeferredValue<LinkedHashMultimap> {
    private LinkedHashMultimapBuffer(int size) {
      super(size);
    }

    @Override
    public LinkedHashMultimap call() {
      int totalValues = 0;
      for (int i = 0; i < size(); i++) {
        totalValues += values[i].length;
      }
      LinkedHashMultimap result = LinkedHashMultimap.create(size(), totalValues / size());
      for (int i = 0; i < size(); i++) {
        result.putAll(keys[i], Arrays.asList(values[i]));
      }
      return result;
    }
  }

  private MultimapCodecs() {}
}
