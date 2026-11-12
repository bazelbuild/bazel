// Copyright 2026 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableSet;
import com.google.errorprone.annotations.Keep;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import net.starlark.java.eval.StarlarkSet;

/**
 * A canonicalizing codec for {@link StarlarkSet} instances.
 *
 * <p>Serializes only the elements in iteration order (requiring the set to be frozen), and
 * deserializes into an immutable {@link ImmutableSet}-backed {@link StarlarkSet}, matching the
 * representation produced by {@link StarlarkSet#unsafeOptimizeMemoryLayout}.
 *
 * <p>Since set elements must be hashable and sets are not, a set cannot (transitively) contain
 * itself, so deferred construction after all elements are available is safe.
 */
@Keep
final class StarlarkSetCodec extends DeferredObjectCodec<StarlarkSet<Object>> {
  // Class literals cannot be parameterized. The cast is safe because the codec handles all
  // StarlarkSet instances, treating their elements as Objects.
  @Override
  @SuppressWarnings("unchecked")
  public Class<StarlarkSet<Object>> getEncodedClass() {
    return (Class<StarlarkSet<Object>>) (Class<?>) StarlarkSet.class;
  }

  @Override
  public void serialize(
      SerializationContext context, StarlarkSet<Object> set, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    if (!set.isImmutable()) {
      throw new SerializationException("Trying to serialize unfrozen StarlarkSet: " + set);
    }
    // Serialize from a single array snapshot so that a concurrent unsafeOptimizeMemoryLayout()
    // cannot cause the size and elements to be read from different backing sets.
    Object[] elements = set.toArray();
    codedOut.writeInt32NoTag(elements.length);
    for (Object element : elements) {
      context.serialize(element, codedOut);
    }
  }

  @Override
  public DeferredValue<StarlarkSet<Object>> deserializeDeferred(
      AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int size = codedIn.readInt32();
    if (size < 0) {
      throw new SerializationException("Expected non-negative size: " + size);
    }
    Object[] elements = new Object[size];
    deserializeObjectArray(context, codedIn, elements, size);
    return () -> StarlarkSet.immutableCopyOf(ImmutableSet.copyOf(elements));
  }
}
