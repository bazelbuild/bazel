// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.skyframe.serialization.ExampleKey.exampleKeyCodec;

import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec.DeferredValue;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/** Class deserialized using {@link DeserializationContext#getSkyValue}. */
record ExampleValue(ExampleKey key, int x) implements SkyValue {
  static ExampleValueCodec exampleValueCodec() {
    return ExampleValueCodec.INSTANCE;
  }

  private static final class ExampleValueCodec extends DeferredObjectCodec<ExampleValue> {
    private static final ExampleValueCodec INSTANCE = new ExampleValueCodec();

    @Override
    public Class<ExampleValue> getEncodedClass() {
      return ExampleValue.class;
    }

    @Override
    public void serialize(
        SerializationContext context, ExampleValue obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serializeLeaf(obj.key(), exampleKeyCodec(), codedOut);
    }

    @Override
    public DeferredValue<ExampleValue> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      ExampleKey key = context.deserializeLeaf(codedIn, exampleKeyCodec());
      SimpleDeferredValue<ExampleValue> builder = SimpleDeferredValue.create();
      context.getSkyValue(key, builder, SimpleDeferredValue::set);
      return builder;
    }
  }
}
