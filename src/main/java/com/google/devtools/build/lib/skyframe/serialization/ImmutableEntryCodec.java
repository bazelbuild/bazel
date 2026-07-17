// Copyright 2025 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.Maps.immutableEntry;

import com.google.common.collect.Maps;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.Map;

/** A codec for the hidden {@link Map.Entry} subclass emitted by {@link Maps#immutableEntry}. */
@SuppressWarnings("rawtypes")
class ImmutableEntryCodec extends DeferredObjectCodec<Map.Entry> {
  private static final Class<? extends Map.Entry> IMMUTABLE_ENTRY_TYPE =
      immutableEntry("", "").getClass();

  @Override
  public Class<? extends Map.Entry> getEncodedClass() {
    return IMMUTABLE_ENTRY_TYPE;
  }

  @Override
  public void serialize(SerializationContext context, Map.Entry entry, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    context.serialize(entry.getKey(), codedOut);
    context.serialize(entry.getValue(), codedOut);
  }

  @Override
  public DeferredValue<Map.Entry> deserializeDeferred(
      AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    var result = new EntryBuilder();
    context.deserialize(codedIn, result, EntryBuilder::setKey);
    context.deserialize(codedIn, result, EntryBuilder::setValue);
    return result;
  }

  private static class EntryBuilder implements DeferredValue<Map.Entry> {
    private Object key;
    private Object value;

    @Override
    public Map.Entry call() {
      return immutableEntry(key, value);
    }

    private static final void setKey(EntryBuilder builder, Object key) {
      builder.key = key;
    }

    private static final void setValue(EntryBuilder builder, Object value) {
      builder.value = value;
    }
  }
}
