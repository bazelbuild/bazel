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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.AbstractMap;
import java.util.Map;
import java.util.function.BiFunction;

@SuppressWarnings("rawtypes")
class MapEntryCodec implements ObjectCodec<Map.Entry> {

  private final BiFunction<Object, Object, Map.Entry> factory;
  private final Class<? extends Map.Entry> type;

  private MapEntryCodec(BiFunction<Object, Object, Map.Entry> factory) {
    this.factory = factory;
    this.type = factory.apply(0, 0).getClass();
  }

  @Override
  public Class<? extends Map.Entry> getEncodedClass() {
    return type;
  }

  @Override
  public void serialize(SerializationContext context, Map.Entry entry, CodedOutputStream codedOut)
      throws IOException, SerializationException {
    context.serialize(entry.getKey(), codedOut);
    context.serialize(entry.getValue(), codedOut);
  }

  @Override
  public Map.Entry deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    return factory.apply(context.deserialize(codedIn), context.deserialize(codedIn));
  }

  private static class MapEntryCodecRegisterer implements CodecRegisterer<MapEntryCodec> {
    @Override
    public Iterable<MapEntryCodec> getCodecsToRegister() {
      return ImmutableList.of(
          new MapEntryCodec(Maps::immutableEntry),
          new MapEntryCodec(AbstractMap.SimpleEntry::new),
          new MapEntryCodec(AbstractMap.SimpleImmutableEntry::new));
    }
  }
}
