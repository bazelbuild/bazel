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

@SuppressWarnings("rawtypes")
class EnumRuntimeCodec implements ObjectCodec<Enum> {

  @Override
  public Class<Enum> getEncodedClass() {
    return Enum.class;
  }

  @Override
  public void serialize(SerializationContext context, Enum value, CodedOutputStream codedOut)
      throws IOException, SerializationException {
    context.serialize(value.getDeclaringClass(), codedOut);
    codedOut.writeInt32NoTag(value.ordinal());
  }

  @Override
  public Enum deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    Class<? extends Enum> enumType = context.deserialize(codedIn);
    return enumType.getEnumConstants()[codedIn.readInt32()];
  }
}
