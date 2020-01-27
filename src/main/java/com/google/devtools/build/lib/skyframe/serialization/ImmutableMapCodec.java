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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Ordering;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.Comparator;
import java.util.Map;

/**
 * Encodes an {@link ImmutableMap}, which may be an {@link ImmutableSortedMap}. The iteration order
 * of the deserialized map is the same as the original map's.
 *
 * <p>We handle both {@link ImmutableMap} and {@link ImmutableSortedMap} here because we cannot
 * handle {@link ImmutableSortedMap} with any ordering other than the default, and so we degrade to
 * ImmutableMap in that case, hoping that the caller does not notice. Since the ordering is
 * preserved, the caller should not be sensitive to this change unless the caller's field is
 * declared to be an {@link ImmutableSortedMap}, in which case we will crash at runtime.
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
class ImmutableMapCodec<V> implements ObjectCodec<ImmutableMap<?, V>> {
  @SuppressWarnings("unchecked")
  @Override
  public Class<ImmutableMap<?, V>> getEncodedClass() {
    // Because Java disallows converting from Class<ImmutableMap> to Class<ImmutableMap<?, ?>>
    // directly.
    return (Class<ImmutableMap<?, V>>) ((Class<?>) ImmutableMap.class);
  }

  @Override
  public void serialize(
      SerializationContext context, ImmutableMap<?, V> map, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    codedOut.writeInt32NoTag(map.size());
    boolean serializeAsSortedMap = false;
    if (map instanceof ImmutableSortedMap) {
      Comparator<?> comparator = ((ImmutableSortedMap<?, ?>) map).comparator();
      // In practice the comparator seems to always be Ordering.natural(), but be flexible.
      serializeAsSortedMap =
          comparator.equals(Ordering.natural()) || comparator.equals(Comparator.naturalOrder());
    }
    codedOut.writeBoolNoTag(serializeAsSortedMap);
    serializeEntries(context, map.entrySet(), codedOut);
  }

  static <K, V> void serializeEntries(
      SerializationContext context,
      Iterable<? extends Map.Entry<K, V>> entrySet,
      CodedOutputStream codedOut)
      throws IOException, SerializationException {
    for (Map.Entry<?, ?> entry : entrySet) {
      context.serialize(entry.getKey(), codedOut);
      try {
        context.serialize(entry.getValue(), codedOut);
      } catch (SerializationException | IOException e) {
        throw SerializationException.propagate(
            String.format(
                "Exception while serializing value of type %s for key '%s'",
                entry.getValue().getClass().getName(), entry.getKey()),
            e);
      }
    }
  }

  @Override
  public ImmutableMap<?, V> deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int length = codedIn.readInt32();
    if (length < 0) {
      throw new SerializationException("Expected non-negative length: " + length);
    }
    ImmutableMap.Builder<?, V> builder;
    if (codedIn.readBool()) {
      builder = deserializeEntries(ImmutableSortedMap.naturalOrder(), length, context, codedIn);
    } else {
      builder =
          deserializeEntries(
              ImmutableMap.builderWithExpectedSize(length), length, context, codedIn);
    }
    try {
      return builder.build();
    } catch (IllegalArgumentException e) {
      throw new SerializationException(
          "Duplicate keys during ImmutableMapCodec deserialization", e);
    }
  }

  static <K, V, M extends ImmutableMap.Builder<K, V>> M deserializeEntries(
      M builder, int length, DeserializationContext context, CodedInputStream codedIn)
      throws IOException, SerializationException {
    for (int i = 0; i < length; i++) {
      K key = context.deserialize(codedIn);
      V value;
      try {
        value = context.deserialize(codedIn);
      } catch (SerializationException | IOException e) {
        throw SerializationException.propagate(
            String.format("Exception while deserializing value for key '%s'", key), e);
      }
      builder.put(key, value);
    }
    return builder;
  }
}
