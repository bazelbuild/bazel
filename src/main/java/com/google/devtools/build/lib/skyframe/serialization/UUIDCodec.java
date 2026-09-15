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

import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.UUID;

class UUIDCodec extends LeafObjectCodec<UUID> {

  @Override
  public Class<UUID> getEncodedClass() {
    return UUID.class;
  }

  @Override
  public void serialize(LeafSerializationContext context, UUID uuid, CodedOutputStream codedOut)
      throws IOException {
    codedOut.writeInt64NoTag(uuid.getMostSignificantBits());
    codedOut.writeInt64NoTag(uuid.getLeastSignificantBits());
  }

  @Override
  public UUID deserialize(LeafDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    return new UUID(codedIn.readInt64(), codedIn.readInt64());
  }
}
