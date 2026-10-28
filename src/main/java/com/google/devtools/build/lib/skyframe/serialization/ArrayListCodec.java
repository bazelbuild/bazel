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
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.ArrayList;

/**
 * Codec for {@link ArrayList}.
 *
 * <p>This is needed because {@link ArrayList} marks its {@code elementData} field transient and
 * even if it weren't, it uses an array slightly larger than its size.
 */
@SuppressWarnings({"rawtypes", "NonApiType"})
class ArrayListCodec extends AsyncObjectCodec<ArrayList> {
  /**
   * Handles to the private {@link ArrayList} fields, used to make a list that wraps an array.
   *
   * <p>This requires {@code java.base/java.util} to be opened to this module. These fields aren't
   * final, so writing to them is unaffected by final field mutation restrictions (JEP 500).
   */
  private static final VarHandle ELEMENT_DATA;

  private static final VarHandle SIZE;

  static {
    try {
      MethodHandles.Lookup lookup =
          MethodHandles.privateLookupIn(ArrayList.class, MethodHandles.lookup());
      ELEMENT_DATA = lookup.findVarHandle(ArrayList.class, "elementData", Object[].class);
      SIZE = lookup.findVarHandle(ArrayList.class, "size", int.class);
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  @Override
  public Class<ArrayList> getEncodedClass() {
    return ArrayList.class;
  }

  @Override
  public void serialize(SerializationContext context, ArrayList list, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    codedOut.writeInt32NoTag(list.size());
    for (Object item : list) {
      context.serialize(item, codedOut);
    }
  }

  @Override
  public ArrayList deserializeAsync(AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int length = codedIn.readInt32();
    if (length < 0) {
      throw new SerializationException("Expected non-negative length: " + length);
    }

    // Assigning anything other than 0 causes ArrayList's constructor to initialize an array, which
    // just gets discarded below when elementData is explicitly set using its VarHandle.
    ArrayList list = new ArrayList(/* initialCapacity= */ 0);
    if (length == 0) {
      context.registerInitialValue(list);
      return list;
    }

    // Makes the list wrap `elements` directly, with its final size, before registering it. Elements
    // are then written into the array as they become available, possibly concurrently.
    Object[] elements = new Object[length];
    ELEMENT_DATA.set(list, elements);
    SIZE.set(list, length);
    context.registerInitialValue(list);

    for (int i = 0; i < length; i++) {
      context.deserializeArrayElement(codedIn, elements, i);
    }

    return list;
  }
}
