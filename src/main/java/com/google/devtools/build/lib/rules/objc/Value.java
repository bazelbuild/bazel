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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;

/**
 * Represents a type whose equality, hash code, and string representation are defined by a single
 * immutable array. This class is designed to be extended by a final class, and to pass the member
 * data to this class's constructor.
 *
 * @param <V> the base class that extends {@code Value}
 */
// TODO(bazel-team): Replace with AutoValue once that is supported in bazel.
public class Value<V extends Value<V>> {
  private final Object memberData;

  /**
   * Constructs a new instance with the given member data. Generally, all member data should be
   * reflected in final fields in the child class.
   * @throws NullPointerException if any element in {@code memberData} is null
   */
  public Value(Object... memberData) {
    Preconditions.checkArgument(memberData.length > 0);
    this.memberData = (memberData.length == 1)
        ? Preconditions.checkNotNull(memberData[0]) : ImmutableList.copyOf(memberData);
  }

  /**
   * A type-safe alternative to calling {@code a.equals(b)}. When using {@code a.equals(b)},
   * {@code b} may accidentally be a different class from {@code a}, in which case there will be no
   * compiler warning and the result will always be false. This method requires both values to have
   * compatible types and to be non-null.
   */
  public boolean equalsOther(V other) {
    return equals(Preconditions.checkNotNull(other));
  }

  @Override
  public boolean equals(Object o) {
    if ((o == null) || (o.getClass() != getClass())) {
      return false;
    }
    Value<?> other = (Value<?>) o;
    return memberData.equals(other.memberData);
  }

  @Override
  public int hashCode() {
    return memberData.hashCode();
  }

  @Override
  public String toString() {
    return getClass().getSimpleName() + ":" + memberData.toString();
  }
}
