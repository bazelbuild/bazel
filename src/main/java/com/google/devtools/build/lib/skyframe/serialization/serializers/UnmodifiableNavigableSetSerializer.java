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

package com.google.devtools.build.lib.skyframe.serialization.serializers;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Sets;
import java.lang.reflect.Field;
import java.util.NavigableSet;
import java.util.TreeSet;

/** A {@link Serializer} for {@link ImmutableSortedSet}. */
class UnmodifiableNavigableSetSerializer extends Serializer<NavigableSet<?>> {

  private final Field delegate;

  private UnmodifiableNavigableSetSerializer() {
    setImmutable(true);
    try {
      Class<?> clazz = Class.forName(Sets.class.getCanonicalName() + "$UnmodifiableNavigableSet");
      delegate = clazz.getDeclaredField("delegate");
      delegate.setAccessible(true);
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException("Issues reflectively writing UnmodifiableNavigableSet", e);
    }
  }

  private Object getDelegateFromUnmodifiableNavigableSet(NavigableSet<?> object) {
    try {
      return delegate.get(object);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException("Issues reflectively writing UnmodifiableNavigableSet", e);
    }
  }

  @Override
  public void write(Kryo kryo, Output output, NavigableSet<?> object) {
    // We want to preserve the underlying delegate class, so we need to reflectively get it and
    // write it directly via kryo
    kryo.writeClassAndObject(output, getDelegateFromUnmodifiableNavigableSet(object));
  }

  @Override
  public NavigableSet<?> read(Kryo kryo, Input input, Class<NavigableSet<?>> type) {
    return Sets.unmodifiableNavigableSet((NavigableSet<?>) kryo.readClassAndObject(input));
  }

  static void registerSerializers(Kryo kryo) {
    kryo.register(getSerializedClass(), new UnmodifiableNavigableSetSerializer());
  }

  @VisibleForTesting
  static @SuppressWarnings("rawtypes") Class<? extends NavigableSet> getSerializedClass() {
    return Sets.unmodifiableNavigableSet(new TreeSet<Object>()).getClass();
  }
}
