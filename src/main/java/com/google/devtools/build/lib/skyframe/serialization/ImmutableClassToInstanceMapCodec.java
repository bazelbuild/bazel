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

import static com.google.devtools.build.lib.skyframe.serialization.MapHelpers.deserializeMapEntries;
import static com.google.devtools.build.lib.skyframe.serialization.MapHelpers.serializeMapEntries;

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
@SuppressWarnings({"unchecked", "rawtypes"})
class ImmutableClassToInstanceMapCodec extends DeferredObjectCodec<ImmutableClassToInstanceMap> {

  @Override
  public Class<ImmutableClassToInstanceMap> getEncodedClass() {
    return ImmutableClassToInstanceMap.class;
  }

  @Override
  public void serialize(
      SerializationContext context, ImmutableClassToInstanceMap map, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    codedOut.writeInt32NoTag(map.size());
    serializeMapEntries(context, map, codedOut);
  }

  @Override
  public DeferredValue<ImmutableClassToInstanceMap> deserializeDeferred(
      AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int size = codedIn.readInt32();
    if (size < 0) {
      throw new SerializationException("Expected non-negative length: " + size);
    }
    if (size == 0) {
      return ImmutableClassToInstanceMap::of;
    }

    EntryBuffer buffer = new EntryBuffer(size);
    deserializeMapEntries(
        context,
        codedIn,
        buffer.keys,
        buffer.values);
    return buffer;
  }

  private static class EntryBuffer implements DeferredValue<ImmutableClassToInstanceMap> {
    final Object[] keys;
    final Object[] values;

    private EntryBuffer(int size) {
      this.keys = new Object[size];
      this.values = new Object[size];
    }

    @Override
    public ImmutableClassToInstanceMap call() {
      ImmutableClassToInstanceMap.Builder builder = ImmutableClassToInstanceMap.builder();
      for (int i = 0; i < size(); i++) {
        builder.put((Class) keys[i], values[i]);
      }
      return builder.build();
    }

    private int size() {
      return keys.length;
    }
  }
}
