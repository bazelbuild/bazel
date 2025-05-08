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

import static com.google.common.base.Preconditions.checkState;
import static com.google.devtools.build.lib.collect.nestedset.NestedArrayCodec.nestedArrayCodec;

import com.google.devtools.build.lib.skyframe.serialization.AsyncDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/**
 * A codec implementation that is asynchronous-compatible.
 *
 * <p>This is required if deserialization of the {@link NestedSet} elements may perform Skyframe
 * lookups using {@link AsyncDeserializationContext#getSkyValue}.
 */
public final class DeferredNestedSetCodec extends DeferredObjectCodec<NestedSet<?>> {

  @Override
  public boolean autoRegister() {
    return false;
  }

  @Override
  @SuppressWarnings("unchecked")
  public Class<NestedSet<?>> getEncodedClass() {
    return (Class<NestedSet<?>>) ((Class<?>) NestedSet.class);
  }

  @Override
  public void serialize(SerializationContext context, NestedSet<?> obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    checkState(!obj.isEmpty(), "empty NestedSet should have been a serialization constant");
    codedOut.writeInt32NoTag(obj.getDepthAndOrder());
    if (obj.isSingleton()) {
      codedOut.writeBoolNoTag(true);
      context.serialize(obj.getChildren(), codedOut);
      return;
    }
    codedOut.writeBoolNoTag(false);
    context.putSharedValue(
        (Object[]) obj.getChildren(), /* distinguisher= */ null, nestedArrayCodec(), codedOut);
  }

  @Override
  public DeferredValue<NestedSet<?>> deserializeDeferred(
      AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    int depthAndOrder = codedIn.readInt32();
    Order order = Order.values()[depthAndOrder & 3];
    int depth = depthAndOrder >> 2;
    var builder = new DeserializationBuilder(order, depth);
    if (codedIn.readBool()) { // singleton
      context.deserialize(codedIn, builder, DeserializationBuilder::setChildren);
    } else {
      context.getSharedValue(
          codedIn,
          /* distinguisher= */ null,
          nestedArrayCodec(),
          builder,
          DeserializationBuilder::setChildren);
    }
    return builder;
  }

  private static class DeserializationBuilder implements DeferredValue<NestedSet<?>> {
    private final Order order;
    private final int approxDepth;
    private Object children;

    private DeserializationBuilder(Order order, int approxDepth) {
      this.order = order;
      this.approxDepth = approxDepth;
    }

    @Override
    public NestedSet<?> call() {
      return NestedSet.forDeserialization(order, approxDepth, children);
    }

    private static void setChildren(DeserializationBuilder builder, Object children) {
      builder.children = children;
    }
  }

  static {
    // A sanity check of for NestedSet.depthAndOrder properties, which this codec depends on.
    checkState(Order.values().length == 4);
  }
}
