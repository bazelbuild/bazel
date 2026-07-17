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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import net.starlark.java.eval.CallUtils;
import net.starlark.java.eval.CallUtils.BuiltinManager;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * A codec for {@link BuiltinManager} that serializes no payload on the wire, and which deserializes
 * by simply requesting the manager for the {@link StarlarkSemantics} that's stored in Skyframe as a
 * precomputed.
 *
 * <p>The {@link BuiltinManager} generally needs to be serialized for the sake of {@link
 * MethodDescriptor}, which itself is referred to by {@link BuiltinFunction}. That is, this is
 * needed when builtin Starlark symbols (e.g. {@code len()}, or .bzl-specific native symbols) get
 * serialized.
 */
public final class BuiltinManagerCodec extends DeferredObjectCodec<BuiltinManager> {

  @Override
  public Class<BuiltinManager> getEncodedClass() {
    return BuiltinManager.class;
  }

  @Override
  public void serialize(
      SerializationContext context, BuiltinManager obj, CodedOutputStream codedOut) {
    // Nothing to serialize.
  }

  @Override
  public DeferredValue<BuiltinManager> deserializeDeferred(
      AsyncDeserializationContext context, CodedInputStream codedIn) throws SerializationException {
    var builder = new DeserializationBuilder();
    context.getSkyValue(
        PrecomputedValue.STARLARK_SEMANTICS.getKey(),
        builder,
        DeserializationBuilder::setStarlarkSemanticsFromPrecomputedValue);

    return builder;
  }

  private static final class DeserializationBuilder implements DeferredValue<BuiltinManager> {
    private StarlarkSemantics semantics;

    @Override
    public BuiltinManager call() {
      checkNotNull(semantics, "StarlarkSemantics not set");
      return CallUtils.getBuiltinManager(semantics);
    }

    private static void setStarlarkSemanticsFromPrecomputedValue(
        DeserializationBuilder builder, Object value) {
      builder.semantics = (StarlarkSemantics) ((PrecomputedValue) value).get();
    }
  }
}
