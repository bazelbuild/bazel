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

import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedHashMap;

/**
 * {@link ObjectCodec} for {@link HashMap} that returns {@link LinkedHashMap} for determinism.
 *
 * <p>This type transformation is safe because {@link LinkedHashMap} is a subclass of {@link
 * HashMap}.
 */
@SuppressWarnings({"unchecked", "rawtypes", "NonApiType"})
final class HashMapCodec extends AsyncObjectCodec<HashMap> {
  @Override
  public Class<HashMap> getEncodedClass() {
    return HashMap.class;
  }

  @Override
  public void serialize(SerializationContext context, HashMap obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    codedOut.writeInt32NoTag(obj.size());
    serializeMapEntries(context, obj, codedOut);
  }

  @Override
  public HashMap deserializeAsync(AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int size = codedIn.readInt32();
    // Load factor is 0.75, so we need an initial capacity of 4/3 actual size to avoid rehashing.
    LinkedHashMap result = new LinkedHashMap(4 * size / 3);

    context.registerInitialValue(result);
    if (size == 0) {
      return result;
    }

    populateMap(context, codedIn, result, size);

    return result;
  }

  static void populateMap(
      AsyncDeserializationContext context, CodedInputStream codedIn, LinkedHashMap map, int size)
      throws SerializationException, IOException {
    EntryBuffer buffer = new EntryBuffer(map, size);
    deserializeMapEntries(
        context, codedIn, buffer.keys, buffer.values, /* done= */ (Runnable) buffer);
  }

  /**
   * Buffers the keys and values until all are available, then populates the map.
   *
   * <p>This approach is thread-safe.
   */
  private static class EntryBuffer implements Runnable {
    private final LinkedHashMap result;
    private final Object[] keys;
    private final Object[] values;

    private EntryBuffer(LinkedHashMap result, int size) {
      this.result = result;
      this.keys = new Object[size];
      this.values = new Object[size];
    }

    @Override
    public void run() {
      for (int i = 0; i < keys.length; i++) {
        result.put(keys[i], values[i]);
      }
    }
  }
}
