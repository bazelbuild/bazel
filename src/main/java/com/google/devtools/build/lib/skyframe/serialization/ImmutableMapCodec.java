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
 * Encodes an {@link ImmutableMap}, which may be an ImmutableSortedMap. We handle both here because
 * we cannot handle ImmutableSortedMap with any ordering other than the default, and so we degrade
 * to ImmutableMap in that case, hoping that the caller does not notice.
 */
class ImmutableMapCodec implements ObjectCodec<ImmutableMap<?, ?>> {
  @SuppressWarnings("unchecked")
  @Override
  public Class<ImmutableMap<?, ?>> getEncodedClass() {
    // Because Java disallows converting from Class<ImmutableMap> to Class<ImmutableMap<?, ?>>
    // directly.
    return (Class<ImmutableMap<?, ?>>) ((Class<?>) ImmutableMap.class);
  }

  @Override
  public void serialize(
      SerializationContext context, ImmutableMap<?, ?> map, CodedOutputStream codedOut)
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
    for (Map.Entry<?, ?> entry : map.entrySet()) {
      context.serialize(entry.getKey(), codedOut);
      context.serialize(entry.getValue(), codedOut);
    }
  }

  @Override
  public ImmutableMap<?, ?> deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int length = codedIn.readInt32();
    if (length < 0) {
      throw new SerializationException("Expected non-negative length: " + length);
    }
    if (codedIn.readBool()) {
      return buildMap(ImmutableSortedMap.naturalOrder(), length, context, codedIn);
    } else {
      return buildMap(ImmutableMap.builderWithExpectedSize(length), length, context, codedIn);
    }
  }

  private static <T> ImmutableMap<T, Object> buildMap(
      ImmutableMap.Builder<T, Object> builder,
      int length,
      DeserializationContext context,
      CodedInputStream codedIn)
      throws IOException, SerializationException {
    for (int i = 0; i < length; i++) {
      T key = context.deserialize(codedIn);
      Object value = context.deserialize(codedIn);
      builder.put(key, value);
    }
    try {
      return builder.build();
    } catch (IllegalArgumentException e) {
      throw new SerializationException(
          "Duplicate keys during ImmutableMapCodec deserialization", e);
    }
  }
}
