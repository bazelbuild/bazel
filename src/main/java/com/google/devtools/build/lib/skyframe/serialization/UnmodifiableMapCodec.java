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

import static com.google.devtools.build.lib.skyframe.serialization.HashMapCodec.populateMap;
import static com.google.devtools.build.lib.skyframe.serialization.MapHelpers.serializeMapEntries;

import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/** {@link ObjectCodec} for {@link java.util.Collections.UnmodifiableMap}. */
@SuppressWarnings({"unchecked", "rawtypes", "NonApiType"})
final class UnmodifiableMapCodec extends AsyncObjectCodec<Map> {
  @SuppressWarnings("ConstantCaseForConstants")
  private static final Map EMPTY = Collections.unmodifiableMap(new HashMap());

  @Override
  public Class getEncodedClass() {
    return EMPTY.getClass();
  }

  @Override
  public void serialize(SerializationContext context, Map obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    codedOut.writeInt32NoTag(obj.size());
    serializeMapEntries(context, obj, codedOut);
  }

  @Override
  public Map deserializeAsync(AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int size = codedIn.readInt32();
    if (size == 0) {
      return EMPTY;
    }

    // Load factor is 0.75, so we need an initial capacity of 4/3 actual size to avoid rehashing.
    LinkedHashMap map = new LinkedHashMap(4 * size / 3);

    Map result = Collections.unmodifiableMap(map);
    context.registerInitialValue(result);

    populateMap(context, codedIn, map, size);
    return result;
  }
}
