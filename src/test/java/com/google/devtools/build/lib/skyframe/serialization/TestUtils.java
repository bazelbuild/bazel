// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

/** Helpers for serialization tests. */
class TestUtils {

  private TestUtils() {}

  /** Serialize a value to a new byte array. */
  static <T> byte[] toBytes(ObjectCodec<T> codec, T value)
      throws IOException, SerializationException {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(bytes);
    codec.serialize(value, codedOut);
    codedOut.flush();
    return bytes.toByteArray();
  }

  /** Deserialize a value from a byte array. */
  static <T> T fromBytes(ObjectCodec<T> codec, byte[] bytes)
      throws SerializationException, IOException {
    return codec.deserialize(CodedInputStream.newInstance(bytes));
  }
}
