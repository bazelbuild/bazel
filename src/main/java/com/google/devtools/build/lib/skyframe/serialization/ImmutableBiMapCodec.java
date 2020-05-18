// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableBiMap;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/**
 * Encodes an {@link ImmutableBiMap}. The iteration order of the deserialized map is the same as the
 * original map's.
 *
 * <p>We handle {@link ImmutableBiMap} by treating it as an {@link
 * com/google/devtools/build/lib/skyframe/serialization/ImmutableBiMapCodec.java used only in
 * javadoc: com.google.common.collect.ImmutableMap} and calling the proper conversion method ({@link
 * ImmutableBiMap#copyOf}) when deserializing. This is valid because every {@link ImmutableBiMap} is
 * also an {@link com.google.common.collect.ImmutableMap}.
 *
 * <p>Any {@link SerializationException} or {@link IOException} that arises while serializing or
 * deserializing a map entry's value (not its key) will be wrapped in a new {@link
 * SerializationException} using {@link SerializationException#propagate}. (Note that this preserves
 * the type of {@link SerializationException.NoCodecException} exceptions.) The message will include
 * the {@code toString()} of the entry's key. For errors that occur while serializing, it will also
 * include the class name of the entry's value. Errors that occur while serializing an entry key are
 * not affected.
 */
class ImmutableBiMapCodec<K, V> implements ObjectCodec<ImmutableBiMap<K, V>> {

  @SuppressWarnings("unchecked")
  @Override
  public Class<ImmutableBiMap<K, V>> getEncodedClass() {
    return (Class<ImmutableBiMap<K, V>>) ((Class<?>) ImmutableBiMap.class);
  }

  @Override
  public void serialize(
      SerializationContext context, ImmutableBiMap<K, V> map, CodedOutputStream codedOut)
      throws SerializationException, IOException {

    codedOut.writeInt32NoTag(map.size());
    ImmutableMapCodec.serializeEntries(context, map.entrySet(), codedOut);
  }

  @Override
  public ImmutableBiMap<K, V> deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {

    int length = codedIn.readInt32();
    if (length < 0) {
      throw new SerializationException("Expected non-negative length: " + length);
    }

    ImmutableBiMap.Builder<K, V> builder =
        ImmutableMapCodec.deserializeEntries(
            ImmutableBiMap.builderWithExpectedSize(length), length, context, codedIn);

    try {
      return builder.build();
    } catch (IllegalArgumentException e) {
      throw new SerializationException(
          "Duplicate keys during ImmutableBiMapCodec deserialization", e);
    }
  }
}
