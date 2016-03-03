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
package com.google.devtools.build.skyframe;

import com.google.common.base.Function;
import com.google.common.collect.Interner;
import com.google.common.collect.Interners;
import com.google.devtools.build.lib.util.Preconditions;

import java.io.Serializable;

/**
 * A {@link SkyKey} is effectively a pair (type, name) that identifies a Skyframe value.
 */
public final class SkyKey implements Serializable {
  private static final Interner<SkyKey> SKY_KEY_INTERNER = Interners.newWeakInterner();

  private final SkyFunctionName functionName;

  /**
   * The name of the value.
   *
   * <p>This is deliberately an untyped Object so that we can use arbitrary value types (e.g.,
   * Labels, PathFragments, BuildConfigurations, etc.) as value names without incurring
   * serialization costs in the in-memory implementation of the graph.
   */
  private final Object argument;

  /**
   * Cache the hash code for this object. It might be expensive to compute. It is transient because
   * argument's hash code might not be stable across JVM instances.
   */
  private transient int hashCode;

  private SkyKey(SkyFunctionName functionName, Object argument) {
    this.functionName = Preconditions.checkNotNull(functionName);
    this.argument = Preconditions.checkNotNull(argument);
    // 'hashCode' is non-volatile and non-final, so this write may in fact *not* be visible to other
    // threads. But this isn't a concern from a correctness perspective. See the comments in
    // #hashCode for more details.
    this.hashCode = computeHashCode();
  }

  public static SkyKey create(SkyFunctionName functionName, Object argument) {
    // Intern to save memory.
    return SKY_KEY_INTERNER.intern(new SkyKey(functionName, argument));
  }

  public SkyFunctionName functionName() {
    return functionName;
  }

  public Object argument() {
    return argument;
  }

  @Override
  public String toString() {
    return functionName + ":" + argument;
  }

  @Override
  public int hashCode() {
    // We use the hash code caching strategy employed by java.lang.String. There are three subtle
    // things going on here:
    //
    // (1) We use a value of 0 to indicate that the hash code hasn't been computed and cached yet.
    // Yes, this means that if the hash code is really 0 then we will "recompute" it each time. But
    // this isn't a problem in practice since a hash code of 0 should be rare.
    //
    // (2) Since we have no synchronization, multiple threads can race here thinking there are the
    // first one to compute and cache the hash code.
    //
    // (3) Moreover, since 'hashCode' is non-volatile, the cached hash code value written from one
    // thread may not be visible by another.
    //
    // All three of these issues are benign from a correctness perspective; in the end we have no
    // overhead from synchronization, at the cost of potentially computing the hash code more than
    // once.
    int h = hashCode;
    if (h == 0) {
      h = computeHashCode();
      hashCode = h;
    }
    return h;
  }

  private int computeHashCode() {
    return 31 * functionName.hashCode() + argument.hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (getClass() != obj.getClass()) {
      return false;
    }
    SkyKey other = (SkyKey) obj;
    return functionName.equals(other.functionName) && argument.equals(other.argument);
  }

  public static final Function<SkyKey, Object> NODE_NAME = new Function<SkyKey, Object>() {
    @Override
    public Object apply(SkyKey input) {
      return input.argument();
    }
  };
}
