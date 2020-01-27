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

import com.google.common.collect.ImmutableTable;
import com.google.common.collect.Table.Cell;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.Set;

/** Codec for {@link ImmutableTable}. */
public class ImmutableTableCodec<R, C, V> implements ObjectCodec<ImmutableTable<R, C, V>> {

  @SuppressWarnings("unchecked")
  @Override
  public Class<ImmutableTable<R, C, V>> getEncodedClass() {
    // Compiler doesn't like to do a direct cast.
    return (Class<ImmutableTable<R, C, V>>) ((Class<?>) ImmutableTable.class);
  }

  @Override
  public void serialize(
      SerializationContext context, ImmutableTable<R, C, V> object, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    Set<Cell<R, C, V>> cellSet = object.cellSet();
    codedOut.writeInt32NoTag(cellSet.size());
    for (Cell<R, C, V> cell : cellSet) {
      context.serialize(cell.getRowKey(), codedOut);
      context.serialize(cell.getColumnKey(), codedOut);
      context.serialize(cell.getValue(), codedOut);
    }
  }

  @Override
  public ImmutableTable<R, C, V> deserialize(
      DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int length = codedIn.readInt32();
    if (length < 0) {
      throw new SerializationException("Expected non-negative length: " + length);
    }
    ImmutableTable.Builder<R, C, V> builder = ImmutableTable.builder();
    for (int i = 0; i < length; i++) {
      builder.put(
          /*rowKey=*/ context.deserialize(codedIn),
          /*columnKey=*/ context.deserialize(codedIn),
          /*value=*/ context.deserialize(codedIn));
    }
    return builder.build();
  }
}
