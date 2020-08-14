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

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableMap;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/**
 * Encodes an {@link ImmutableClassToInstanceMap}. The iteration order of the deserialized map is
 * the same as the original map's.
 *
 * <p>We handle {@link ImmutableClassToInstanceMap} by treating it as an {@link ImmutableMap} and
 * calling the proper conversion method ({@link ImmutableClassToInstanceMap#copyOf}) when
 * deserializing.
 *
 * <p>Any {@link SerializationException} or {@link IOException} that arises while serializing or
 * deserializing a map entry's value (not its key) will be wrapped in a new {@link
 * SerializationException} using {@link SerializationException#propagate}. (Note that this preserves
 * the type of {@link SerializationException.NoCodecException} exceptions.) The message will include
 * the {@code toString()} of the entry's key. For errors that occur while serializing, it will also
 * include the class name of the entry's value. Errors that occur while serializing an entry key are
 * not affected.
 */
class ImmutableClassToInstanceMapCodec<B> implements ObjectCodec<ImmutableClassToInstanceMap<B>> {

  @SuppressWarnings("unchecked")
  @Override
  public Class<ImmutableClassToInstanceMap<B>> getEncodedClass() {
    return (Class<ImmutableClassToInstanceMap<B>>) ((Class<?>) ImmutableClassToInstanceMap.class);
  }

  @Override
  public void serialize(
      SerializationContext context, ImmutableClassToInstanceMap<B> map, CodedOutputStream codedOut)
      throws SerializationException, IOException {

    codedOut.writeInt32NoTag(map.size());
    ImmutableMapCodec.serializeEntries(context, map.entrySet(), codedOut);
  }

  @Override
  public ImmutableClassToInstanceMap<B> deserialize(
      DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {

    int length = codedIn.readInt32();
    if (length < 0) {
      throw new SerializationException("Expected non-negative length: " + length);
    }

    ImmutableMap.Builder<Class<? extends B>, B> classKeyedBuilder =
        ImmutableMapCodec.deserializeEntries(ImmutableMap.builder(), length, context, codedIn);

    ImmutableMap<Class<? extends B>, B> classKeyedMap;
    try {
      classKeyedMap = classKeyedBuilder.build();
    } catch (IllegalArgumentException e) {
      throw new SerializationException(
          "Duplicate keys during ImmutableClassToInstanceMapCodec deserialization", e);
    }
    try {
      return ImmutableClassToInstanceMap.copyOf(classKeyedMap);
    } catch (ClassCastException e) {
      throw new SerializationException(
          "Key-value type mismatch during ImmutableClassToInstanceMapCodec deserialization", e);
    }
  }
}
