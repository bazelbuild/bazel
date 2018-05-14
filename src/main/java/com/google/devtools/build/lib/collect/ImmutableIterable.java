// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.util.Iterator;

/**
 * A wrapper that signals the immutability of a certain iterable.
 *
 * <p>Intended for use in scenarios when you have an iterable that is de facto immutable, but is not
 * recognized as such by {@link CollectionUtils#checkImmutable(Iterable)}.
 *
 * <p>Only use this when you know that the contents of the underlying iterable will never change, or
 * you will be setting yourself up for aliasing bugs.
 */
@AutoCodec
public final class ImmutableIterable<T> implements Iterable<T> {

  private final Iterable<T> iterable;

  private ImmutableIterable(Iterable<T> iterable) {
    this.iterable = iterable;
  }

  @Override
  public Iterator<T> iterator() {
    return iterable.iterator();
  }

  /** Creates an {@link ImmutableIterable} instance. */
  // Use a factory method in order to avoid having to specify generic arguments.
  @AutoCodec.Instantiator
  public static <T> ImmutableIterable<T> from(Iterable<T> iterable) {
    return new ImmutableIterable<>(iterable);
  }
}
