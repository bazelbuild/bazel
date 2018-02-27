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
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/**
 * A {@link ObjectCodec} for {@link ImmutableList}.
 *
 * <p>This differs from {@link ImmutableListCodec} in that this uses the runtime type of contained
 * objects for serialization.
 */
@SuppressWarnings("rawtypes") // Intentional erasure of ImmutableList.
class ImmutableListRuntimeCodec implements ObjectCodec<ImmutableList> {

  @Override
  public Class<ImmutableList> getEncodedClass() {
    return ImmutableList.class;
  }

  @Override
  public void serialize(
      SerializationContext context, ImmutableList object, CodedOutputStream codedOut)
      throws IOException, SerializationException {
    codedOut.writeInt32NoTag(object.size());
    for (Object obj : object) {
      context.serialize(obj, codedOut);
    }
  }

  @Override
  public ImmutableList deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws IOException, SerializationException {
    int size = codedIn.readInt32();
    Object[] list = new Object[size];
    for (int i = 0; i < size; ++i) {
      list[i] = context.deserialize(codedIn);
    }
    return ImmutableList.copyOf(list);
  }
}
