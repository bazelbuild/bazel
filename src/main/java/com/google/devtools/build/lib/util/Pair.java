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
package com.google.devtools.build.lib.util;

import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.util.Objects;
import javax.annotation.Nullable;

/** An immutable, semantic-free ordered pair of nullable values. Avoid using it in public APIs. */
@AutoCodec
public final class Pair<A, B> {

  /**
   * Creates a new pair containing the given elements in order.
   */
  public static <A, B> Pair<A, B> of(@Nullable A first, @Nullable B second) {
    return new Pair<>(first, second);
  }

  /**
   * The first element of the pair.
   */
  @Nullable
  public final A first;

  /**
   * The second element of the pair.
   */
  @Nullable
  public final B second;

  /**
   * Constructor.  It is usually easier to call {@link #of}.
   */
  public Pair(@Nullable A first, @Nullable B second) {
    this.first = first;
    this.second = second;
  }

  @Nullable
  public A getFirst() {
    return first;
  }

  @Nullable
  public B getSecond() {
    return second;
  }

  @Override
  public String toString() {
    return "(" + first + ", " + second + ")";
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof Pair)) {
      return false;
    }
    Pair<?, ?> p = (Pair<?, ?>) o;
    return Objects.equals(first, p.first) && Objects.equals(second, p.second);
  }

  @Override
  public int hashCode() {
    int hash1 = first == null ? 0 : first.hashCode();
    int hash2 = second == null ? 0 : second.hashCode();
    return 31 * hash1 + hash2;
  }
}
