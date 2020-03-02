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
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Sets;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/** {@link ObjectCodec} for {@link ImmutableSet} and other sets that should be immutable. */
@SuppressWarnings("rawtypes") // Intentional erasure of ImmutableSet.
class ImmutableSetRuntimeCodec implements ObjectCodec<Set> {
  @SuppressWarnings("unchecked")
  private static final Class<Set> LINKED_HASH_MULTIMAP_CLASS =
      (Class<Set>) LinkedHashMultimap.create(ImmutableMultimap.of("a", "b")).get("a").getClass();

  @SuppressWarnings("unchecked")
  private static final Class<Set> SINGLETON_SET_CLASS =
      (Class<Set>) Collections.singleton("a").getClass();

  @SuppressWarnings("unchecked")
  private static final Class<Set> SUBSET_CLASS =
      (Class<Set>) Iterables.getOnlyElement(Sets.powerSet(ImmutableSet.of())).getClass();

  @SuppressWarnings("unchecked")
  private static final Class<Set> EMPTY_SET_CLASS = (Class<Set>) Collections.emptySet().getClass();

  @Override
  public Class<ImmutableSet> getEncodedClass() {
    return ImmutableSet.class;
  }

  @Override
  public ImmutableList<Class<? extends Set>> additionalEncodedClasses() {
    return ImmutableList.of(
        LINKED_HASH_MULTIMAP_CLASS,
        SINGLETON_SET_CLASS,
        EMPTY_SET_CLASS,
        SUBSET_CLASS,
        HashSet.class);
  }

  @Override
  public void serialize(SerializationContext context, Set object, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    codedOut.writeInt32NoTag(object.size());
    for (Object obj : object) {
      context.serialize(obj, codedOut);
    }
  }

  @SuppressWarnings("unchecked") // Adding object to untyped builder.
  @Override
  public ImmutableSet deserialize(DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int size = codedIn.readInt32();
    ImmutableSet.Builder builder = ImmutableSet.builderWithExpectedSize(size);
    for (int i = 0; i < size; i++) {
      // Don't inline so builder knows this is an object, not an array.
      Object item = context.deserialize(codedIn);
      builder.add(item);
    }
    return builder.build();
  }
}
