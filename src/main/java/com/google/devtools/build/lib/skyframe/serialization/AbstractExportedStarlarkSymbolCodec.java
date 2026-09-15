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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.skyframe.BzlLoadValue.bzlLoadKeyCodec;
import static com.google.devtools.build.lib.skyframe.serialization.strings.UnsafeStringCodec.stringCodec;

import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec.DeferredValue;
import com.google.errorprone.annotations.ForOverride;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import net.starlark.java.eval.Module;

/** Base codec for exported Starlark symbols. */
public abstract class AbstractExportedStarlarkSymbolCodec<T> extends DeferredObjectCodec<T> {
  @ForOverride
  protected abstract BzlLoadValue.Key getBzlLoadKey(T obj);

  @ForOverride
  protected abstract String getExportedName(T obj);

  @Override
  public final void serialize(SerializationContext context, T obj, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    context.serializeLeaf(getBzlLoadKey(obj), bzlLoadKeyCodec(), codedOut);
    context.serializeLeaf(getExportedName(obj), stringCodec(), codedOut);
  }

  @Override
  public final DeferredValue<? extends T> deserializeDeferred(
      AsyncDeserializationContext context, CodedInputStream codedIn)
      throws SerializationException, IOException {
    BzlLoadValue.Key bzlLoadKey = context.deserializeLeaf(codedIn, bzlLoadKeyCodec());
    String name = context.deserializeLeaf(codedIn, stringCodec());

    var builder = new DeserializationBuilder<>(getEncodedClass(), name);
    context.getSkyValue(bzlLoadKey, builder, DeserializationBuilder::setBzlLoadValue);
    return builder;
  }

  private static final class DeserializationBuilder<T> implements DeferredValue<T> {
    private final Class<T> type;
    private final String name;
    private BzlLoadValue loadValue;

    private DeserializationBuilder(Class<T> type, String name) {
      this.type = type;
      this.name = name;
    }

    @Override
    public T call() {
      Module module = checkNotNull(loadValue, "Skyframe lookup value not set").getModule();
      return type.cast(checkNotNull(module.getGlobal(name), "%s not found in %s", name, module));
    }

    private static void setBzlLoadValue(DeserializationBuilder<?> builder, Object value) {
      builder.loadValue = (BzlLoadValue) value;
    }
  }
}
