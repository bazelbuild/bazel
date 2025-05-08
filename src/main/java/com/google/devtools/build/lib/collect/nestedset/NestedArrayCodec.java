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
package com.google.devtools.build.lib.collect.nestedset;

import static com.google.devtools.build.lib.collect.nestedset.NestedSet.EMPTY_CHILDREN;

import com.google.devtools.build.lib.skyframe.serialization.AsyncDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/**
 * Codec that handles nested {@code Object[]} with arbitrary contents.
 *
 * <p>This codec is intended for use with {@link SerializationContext#putSharedValue} and uses
 * {@link SerializationContext#putSharedValue} for subarrays to promoting sharing.
 */
final class NestedArrayCodec extends DeferredObjectCodec<Object[]> {
  private static final NestedArrayCodec INSTANCE = new NestedArrayCodec();

  public static NestedArrayCodec nestedArrayCodec() {
    return INSTANCE;
  }

  @Override
  public boolean autoRegister() {
    return false;
  }

  @Override
  public Class<Object[]> getEncodedClass() {
    return Object[].class;
  }

  @Override
  public void serialize(
      SerializationContext context, Object[] nestedArray, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    int length = nestedArray.length;
    codedOut.writeInt32NoTag(length);
    for (int i = 0; i < length; i++) {
      Object child = nestedArray[i];
      if (child instanceof Object[]) {
        codedOut.writeBoolNoTag(true);
        context.putSharedValue(
            (Object[]) child, /* distinguisher= */ null, /* codec= */ this, codedOut);
      } else {
        codedOut.writeBoolNoTag(false);
        context.serialize(child, codedOut);
      }
    }
  }

  @Override
  public DeferredValue<Object[]> deserializeDeferred(
      AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int length = codedIn.readInt32();
    if (length == 0) {
      return () -> EMPTY_CHILDREN;
    }
    Object[] values = new Object[length];
    for (int i = 0; i < length; i++) {
      if (codedIn.readBool()) {
        context.getSharedValue(
            codedIn, /* distinguisher= */ null, /* codec= */ this, values, new ArrayFieldSetter(i));
      } else {
        context.deserialize(codedIn, values, new ArrayFieldSetter(i));
      }
    }
    return () -> values;
  }

  private NestedArrayCodec() {}

  private static final class ArrayFieldSetter
      implements AsyncDeserializationContext.FieldSetter<Object[]> {
    private final int index;

    private ArrayFieldSetter(int index) {
      this.index = index;
    }

    @Override
    public void set(Object[] array, Object value) {
      array[index] = value;
    }
  }
}
