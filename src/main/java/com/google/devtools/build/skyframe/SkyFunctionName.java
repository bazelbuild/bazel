// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.base.Predicate;

import java.io.Serializable;
import java.util.Set;

/**
 * An identifier for a {@code SkyFunction}.
 */
public final class SkyFunctionName implements Serializable {
  public static SkyFunctionName computed(String name) {
    return new SkyFunctionName(name, true);
  }

  private final String name;
  private final boolean isComputed;

  public SkyFunctionName(String name, boolean isComputed) {
    this.name = name;
    this.isComputed = isComputed;
  }

  @Override
  public String toString() {
    return name;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof SkyFunctionName)) {
      return false;
    }
    SkyFunctionName other = (SkyFunctionName) obj;
    return name.equals(other.name);
  }

  @Override
  public int hashCode() {
    return name.hashCode();
  }

  /**
   * Returns whether the values of this type are computed. The computation of a computed value must
   * be deterministic and may only access requested dependencies.
   */
  public boolean isComputed() {
    return isComputed;
  }

  /**
   * A predicate that returns true for {@link SkyKey}s that have the given {@link SkyFunctionName}.
   */
  public static Predicate<SkyKey> functionIs(final SkyFunctionName functionName) {
    return new Predicate<SkyKey>() {
      @Override
      public boolean apply(SkyKey skyKey) {
        return functionName.equals(skyKey.functionName());
      }
    };
  }

  /**
   * A predicate that returns true for {@link SkyKey}s that have the given {@link SkyFunctionName}.
   */
  public static Predicate<SkyKey> functionIsIn(final Set<SkyFunctionName> functionNames) {
    return new Predicate<SkyKey>() {
      @Override
      public boolean apply(SkyKey skyKey) {
        return functionNames.contains(skyKey.functionName());
      }
    };
  }
}
