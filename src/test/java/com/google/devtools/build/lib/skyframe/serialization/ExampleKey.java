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

import static com.google.devtools.build.lib.skyframe.serialization.strings.UnsafeStringCodec.stringCodec;

import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/** An example {@link SkyKey} for testing serialization involving Skyframe. */
record ExampleKey(String name) implements SkyKey {
  @Override
  public SkyFunctionName functionName() {
    throw new UnsupportedOperationException();
  }

  static ExampleKeyCodec exampleKeyCodec() {
    return ExampleKeyCodec.INSTANCE;
  }

  private static final class ExampleKeyCodec extends LeafObjectCodec<ExampleKey> {
    private static final ExampleKeyCodec INSTANCE = new ExampleKeyCodec();

    @Override
    public Class<ExampleKey> getEncodedClass() {
      return ExampleKey.class;
    }

    @Override
    public void serialize(
        LeafSerializationContext context, ExampleKey key, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serializeLeaf(key.name(), stringCodec(), codedOut);
    }

    @Override
    public ExampleKey deserialize(LeafDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      return new ExampleKey(context.deserializeLeaf(codedIn, stringCodec()));
    }
  }
}
