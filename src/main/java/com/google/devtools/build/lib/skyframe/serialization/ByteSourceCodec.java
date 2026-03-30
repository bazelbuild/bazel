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

import com.google.common.io.ByteSource;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/** Codec for {@link ByteSource} that serializes the byte source as a byte array. */
final class ByteSourceCodec extends LeafObjectCodec<ByteSource> {
  @Override
  public Class<ByteSource> getEncodedClass() {
    return ByteSource.class;
  }

  @Override
  public void serialize(
      LeafSerializationContext context, ByteSource obj, CodedOutputStream codedOut)
      throws IOException {
    codedOut.writeByteArrayNoTag(obj.read());
  }

  @Override
  public ByteSource deserialize(LeafDeserializationContext context, CodedInputStream codedIn)
      throws IOException {
    return ByteSource.wrap(codedIn.readByteArray());
  }

  private ByteSourceCodec() {}
}
