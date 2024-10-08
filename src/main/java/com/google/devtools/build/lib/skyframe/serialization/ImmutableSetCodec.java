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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.Set;

/** {@link ObjectCodec} for {@link ImmutableSet} and other sets that should be immutable. */
@SuppressWarnings({"rawtypes", "unchecked"})
final class ImmutableSetCodec extends DeferredObjectCodec<Set> {
  // Conversion of the types below to ImmutableSet is sound because the underlying types are hidden
  // and only referenceable as the Set type.

  @VisibleForTesting
  static final Class<Set> MULTIMAP_VALUE_SET_CLASS =
      (Class<Set>) LinkedHashMultimap.create(ImmutableMultimap.of("a", "b")).get("a").getClass();

  private static final Class<Set> SINGLETON_SET_CLASS =
      (Class<Set>) Collections.singleton("a").getClass();

  private static final Class<Set> SUBSET_CLASS =
      (Class<Set>) Iterables.getOnlyElement(Sets.powerSet(ImmutableSet.of())).getClass();

  /**
   * Defines a reference constant for {@link Collections#emptySet}.
   *
   * <p>This is done here because we can't add the annotation to the JDK code.
   */
  @SuppressWarnings({"EmptySet", "ConstantCaseForConstants"})
  @SerializationConstant
  static final Set EMPTY_SET = Collections.emptySet();

  @Override
  public Class<ImmutableSet> getEncodedClass() {
    return ImmutableSet.class;
  }

  @Override
  public ImmutableSet<Class<? extends Set>> additionalEncodedClasses() {
    return ImmutableSet.of(MULTIMAP_VALUE_SET_CLASS, SINGLETON_SET_CLASS, SUBSET_CLASS);
  }

  @Override
  public void serialize(SerializationContext context, Set object, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    codedOut.writeInt32NoTag(object.size());
    for (Object obj : object) {
      context.serialize(obj, codedOut);
    }
  }

  @Override
  public DeferredValue<Set> deserializeDeferred(
      AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int size = codedIn.readInt32();

    ElementBuffer buffer = new ElementBuffer(size);
    deserializeObjectArray(context, codedIn, buffer.elements, size);
    return buffer;
  }

  private static class ElementBuffer implements DeferredValue<Set> {
    private final Object[] elements;

    private ElementBuffer(int size) {
      this.elements = new Object[size];
    }

    @Override
    public ImmutableSet call() {
      return ImmutableSet.builderWithExpectedSize(elements.length).add(elements).build();
    }
  }
}
