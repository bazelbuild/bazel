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

import static com.google.devtools.build.lib.skyframe.serialization.MapHelpers.deserializeMapEntries;
import static com.google.devtools.build.lib.skyframe.serialization.MapHelpers.serializeMapEntries;
import static com.google.devtools.build.lib.unsafe.UnsafeProvider.getFieldOffset;
import static java.util.Comparator.naturalOrder;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.errorprone.annotations.Keep;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.Comparator;

/**
 * Encodes an {@link ImmutableMap}, which may be an {@link ImmutableSortedMap}. The iteration order
 * of the deserialized map is the same as the original map's.
 *
 * <p>Any {@link SerializationException} or {@link IOException} that arises while serializing or
 * deserializing a map entry's value (not its key) will be wrapped in a new {@link
 * SerializationException} using {@link SerializationException#propagate}. (Note that this preserves
 * the type of {@link SerializationException.NoCodecException} exceptions.) The message will include
 * the {@code toString()} of the entry's key. For errors that occur while serializing, it will also
 * include the class name of the entry's value. Errors that occur while serializing an entry key are
 * not affected.
 *
 * <p>Because of the ambiguity around the key type (Comparable in the case of {@link
 * ImmutableSortedMap}, arbitrary otherwise, we avoid specifying the key type as a parameter.
 */
@SuppressWarnings({"unchecked", "rawtypes"})
public final class ImmutableMapCodecs {

  @SerializationConstant static final Comparator<?> ORDERING_NATURAL = Ordering.natural();

  // In practice, the natural comparator seems to always be Ordering.natural(), but be flexible.
  @SerializationConstant static final Comparator<?> COMPARATOR_NATURAL_ORDER = naturalOrder();

  public static final ImmutableMapCodec IMMUTABLE_MAP_CODEC = new ImmutableMapCodec();

  @Keep // used reflectively
  private static class ImmutableMapCodec extends DeferredObjectCodec<ImmutableMap> {
    @Override
    public Class<ImmutableMap> getEncodedClass() {
      return ImmutableMap.class;
    }

    @Override
    public void serialize(
        SerializationContext context, ImmutableMap map, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      codedOut.writeInt32NoTag(map.size());
      serializeMapEntries(context, map, codedOut);
    }

    @Override
    public DeferredValue<ImmutableMap> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      int size = codedIn.readInt32();
      if (size < 0) {
        throw new SerializationException("Expected non-negative length: " + size);
      }
      if (size == 0) {
        return ImmutableMap::of;
      }

      ImmutableMapEntryBuffer buffer = new ImmutableMapEntryBuffer(size);
      deserializeMapEntries(
          context,
          codedIn,
          buffer.keys,
          buffer.values);
      return buffer;
    }
  }

  @Keep // used reflectively
  private static class ImmutableSortedMapCodec extends DeferredObjectCodec<ImmutableSortedMap> {
    @Override
    public Class<ImmutableSortedMap> getEncodedClass() {
      return ImmutableSortedMap.class;
    }

    @Override
    public void serialize(
        SerializationContext context, ImmutableSortedMap map, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      codedOut.writeInt32NoTag(map.size());
      if (map.isEmpty()) {
        return;
      }

      context.serialize(map.comparator(), codedOut);
      serializeMapEntries(context, map, codedOut);
    }

    @Override
    public DeferredValue<ImmutableSortedMap> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      int size = codedIn.readInt32();
      if (size < 0) {
        throw new SerializationException("Expected non-negative length: " + size);
      }
      if (size == 0) {
        return ImmutableSortedMap::of;
      }

      ImmutableSortedMapEntryBuffer buffer = new ImmutableSortedMapEntryBuffer(size);
      context.deserialize(codedIn, buffer, COMPARATOR_OFFSET);
      deserializeMapEntries(
          context,
          codedIn,
          buffer.keys,
          buffer.values);
      return buffer;
    }
  }

  private static class EntryBuffer {
    final Object[] keys;
    final Object[] values;

    private EntryBuffer(int size) {
      this.keys = new Object[size];
      this.values = new Object[size];
    }

    int size() {
      return keys.length;
    }
  }

  private static class ImmutableMapEntryBuffer extends EntryBuffer
      implements DeferredObjectCodec.DeferredValue<ImmutableMap> {
    private ImmutableMapEntryBuffer(int size) {
      super(size);
    }

    @Override
    public ImmutableMap call() {
      ImmutableMap.Builder builder = ImmutableMap.builderWithExpectedSize(size());
      for (int i = 0; i < size(); i++) {
        builder.put(keys[i], values[i]);
      }
      return builder.buildOrThrow();
    }
  }

  private static final long COMPARATOR_OFFSET;

  static {
    try {
      COMPARATOR_OFFSET = getFieldOffset(ImmutableSortedMapEntryBuffer.class, "comparator");
    } catch (NoSuchFieldException e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  private static class ImmutableSortedMapEntryBuffer extends EntryBuffer
      implements DeferredObjectCodec.DeferredValue<ImmutableSortedMap> {
    private Comparator comparator;

    private ImmutableSortedMapEntryBuffer(int size) {
      super(size);
    }

    @Override
    public ImmutableSortedMap call() {
      ImmutableSortedMap.Builder builder = new ImmutableSortedMap.Builder(comparator);
      for (int i = 0; i < size(); i++) {
        builder.put(keys[i], values[i]);
      }
      return builder.buildOrThrow();
    }
  }

  private ImmutableMapCodecs() {}
}
