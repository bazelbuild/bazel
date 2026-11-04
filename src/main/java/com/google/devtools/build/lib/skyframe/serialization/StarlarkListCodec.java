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

import static com.google.devtools.build.lib.skyframe.serialization.ArrayProcessor.deserializeObjectArray;

import com.google.errorprone.annotations.Keep;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import net.starlark.java.eval.ImmutableSingletonStarlarkList;
import net.starlark.java.eval.RegularImmutableStarlarkList;
import net.starlark.java.eval.StarlarkList;

/**
 * A canonicalizing codec for {@link StarlarkList} instances.
 *
 * <p>Serializes only the list elements (requiring the list to be frozen), and deserializes into the
 * canonical immutable {@link StarlarkList} representation for its size.
 */
@Keep
final class StarlarkListCodec extends AsyncObjectCodec<StarlarkList<Object>> {
  // Class literals cannot be parameterized. The cast is safe because the codec handles all
  // StarlarkList instances, treating their elements as Objects.
  @Override
  @SuppressWarnings("unchecked")
  public Class<StarlarkList<Object>> getEncodedClass() {
    return (Class<StarlarkList<Object>>) (Class<?>) StarlarkList.class;
  }

  @Override
  public void serialize(
      SerializationContext context, StarlarkList<Object> list, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    if (!list.isImmutable()) {
      throw new SerializationException("Trying to serialize unfrozen StarlarkList: " + list);
    }
    Object[] elems = list.elems();
    // The backing array may have spare capacity beyond size() (e.g. for frozen mutable lists, or
    // until unsafeOptimizeMemoryLayout() trims it), so elems.length must not be used. Only the
    // first size() elements are written, ensuring that the serialized bytes are independent of the
    // list's capacity.
    int size = list.size();
    codedOut.writeInt32NoTag(size);
    for (int i = 0; i < size; i++) {
      context.serialize(elems[i], codedOut);
    }
  }

  @Override
  public StarlarkList<Object> deserializeAsync(
      AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int size = codedIn.readInt32();
    return switch (size) {
      case 0 -> {
        // StarlarkList.empty() is deliberately not a @SerializationConstant: constants are matched
        // by identity, so the shared instance would serialize differently from other empty lists
        // (e.g. frozen mutable lists), breaking canonicalization.
        StarlarkList<Object> empty = StarlarkList.empty();
        context.registerInitialValue(empty);
        yield empty;
      }
      case 1 -> {
        var list = new ImmutableSingletonStarlarkList<Object>(null);
        context.registerInitialValue(list);
        context.deserialize(
            codedIn, list, ImmutableSingletonStarlarkList::setElementForDeserialization);
        yield list;
      }
      default -> {
        Object[] elems = new Object[size];
        var list = new RegularImmutableStarlarkList<Object>(elems);
        context.registerInitialValue(list);
        deserializeObjectArray(context, codedIn, elems, size);
        yield list;
      }
    };
  }
}
