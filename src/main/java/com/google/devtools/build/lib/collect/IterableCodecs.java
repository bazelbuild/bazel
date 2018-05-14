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

package com.google.devtools.build.lib.collect;

import com.google.common.collect.FluentIterable;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

@SuppressWarnings("rawtypes")
class IterableCodecs {
  static class FluentIterableCodec implements ObjectCodec<FluentIterable> {
    @Override
    public Class<FluentIterable> getEncodedClass() {
      return FluentIterable.class;
    }

    @Override
    public void serialize(
        SerializationContext context, FluentIterable obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      IterableCodecs.serialize(context, obj, codedOut);
    }

    @Override
    public FluentIterable deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      return FluentIterable.from(IterableCodecs.deserialize(context, codedIn));
    }
  }

  static class IterablesChainCodec implements ObjectCodec<IterablesChain> {
    @Override
    public Class<IterablesChain> getEncodedClass() {
      return IterablesChain.class;
    }

    @Override
    public void serialize(
        SerializationContext context, IterablesChain obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      IterableCodecs.serialize(context, obj, codedOut);
    }

    @Override
    public IterablesChain deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      return new IterablesChain<>(IterableCodecs.deserialize(context, codedIn));
    }
  }

  static class DedupingIterableCodec implements ObjectCodec<DedupingIterable> {
    @Override
    public Class<DedupingIterable> getEncodedClass() {
      return DedupingIterable.class;
    }

    @Override
    public void serialize(
        SerializationContext context, DedupingIterable obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      IterableCodecs.serialize(context, obj, codedOut);
    }

    @Override
    public DedupingIterable deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      return new DedupingIterable<>(IterableCodecs.deserialize(context, codedIn));
    }
  }

  private static void serialize(
      SerializationContext context, Iterable obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    for (Object elt : obj) {
      context.serialize(elt, codedOut);
    }
    context.serialize(DONE, codedOut);
  }

  private static ImmutableList<Object> deserialize(
      DeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    ImmutableList.Builder<Object> builder = ImmutableList.builder();
    for (Object next = context.deserialize(codedIn);
        next != DONE;
        next = context.deserialize(codedIn)) {
      builder.add(next);
    }
    return builder.build();
  }

  /**
   * An object used to mark the end of the sequence of elements.
   *
   * <p>We use this instead of emitting a count to avoid possibly running deduplication twice.
   */
  @AutoCodec @AutoCodec.VisibleForSerialization static final Object DONE = new Object();
}
