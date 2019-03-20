// Copyright 2019 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.skyframe.serialization;

import com.google.common.collect.ImmutableList;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.TreeMap;

/** {@link ObjectCodec} for {@link HashMap} and {@link Collections.UnmodifiableMap}. */
class TreeMapCodec<K, V> implements ObjectCodec<NavigableMap<K, V>> {
  @SuppressWarnings("unchecked")
  private static final Class<NavigableMap<?, ?>> UNMODIFIABLE_TYPE =
      (Class<NavigableMap<?, ?>>) Collections.unmodifiableNavigableMap(new TreeMap<>()).getClass();

  @SuppressWarnings("unchecked")
  @Override
  public Class<NavigableMap<K, V>> getEncodedClass() {
    return cast(TreeMap.class);
  }

  @Override
  public List<Class<? extends NavigableMap<K, V>>> additionalEncodedClasses() {
    return ImmutableList.of(cast(UNMODIFIABLE_TYPE));
  }

  @Override
  public void serialize(SerializationContext context, NavigableMap<K, V> obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    codedOut.writeInt32NoTag(obj.size());
    for (Map.Entry<K, V> entry : obj.entrySet()) {
      context.serialize(entry.getKey(), codedOut);
      context.serialize(entry.getValue(), codedOut);
    }
  }

  @Override
  public NavigableMap<K, V> deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int size = codedIn.readInt32();
    // Load factor is 0.75, so we need an initial capacity of 4/3 actual size to avoid rehashing.
    NavigableMap<K, V> result = new TreeMap<>();
    for (int i = 0; i < size; i++) {
      result.put(context.deserialize(codedIn), context.deserialize(codedIn));
    }
    return Collections.unmodifiableNavigableMap(result);
  }

  @SuppressWarnings({"unchecked", "rawtypes"}) // Raw Map.
  private Class<NavigableMap<K, V>> cast(Class<? extends NavigableMap> clazz) {
    return (Class<NavigableMap<K, V>>) clazz;
  }
}
