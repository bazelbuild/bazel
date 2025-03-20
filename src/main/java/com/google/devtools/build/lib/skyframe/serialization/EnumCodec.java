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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/** Codec for an enum. */
public class EnumCodec<T extends Enum<T>> extends LeafObjectCodec<T> {

  private final Class<T> enumClass;

  /**
   * A cached copy of T.values(), to avoid allocating an array upon every deserialization operation.
   */
  private final ImmutableList<T> values;

  public EnumCodec(Class<T> enumClass) {
    this.enumClass = enumClass;
    this.values = ImmutableList.copyOf(enumClass.getEnumConstants());
  }

  @Override
  public Class<T> getEncodedClass() {
    return enumClass;
  }

  @Override
  public void serialize(LeafSerializationContext context, T value, CodedOutputStream codedOut)
      throws IOException {
    Preconditions.checkNotNull(value, "Enum value for %s is null", enumClass);
    codedOut.writeEnumNoTag(value.ordinal());
  }

  @Override
  public T deserialize(LeafDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int ordinal = codedIn.readEnum();
    try {
      return values.get(ordinal);
    } catch (ArrayIndexOutOfBoundsException e) {
      throw new SerializationException(
          "Invalid ordinal for " + enumClass.getName() + " enum: " + ordinal, e);
    }
  }
}
