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

import com.google.common.base.Function;
import com.google.common.base.Preconditions;

import java.io.Serializable;

/**
 * A {@link SkyKey} is effectively a pair (type, name) that identifies a Skyframe value.
 */
public final class SkyKey implements Serializable {
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
   * Cache the hash code for this object. It might be expensive to compute.
   */
  private final int hashCode;

  public SkyKey(SkyFunctionName functionName, Object valueName) {
    this.functionName = Preconditions.checkNotNull(functionName);
    this.argument = Preconditions.checkNotNull(valueName);
    this.hashCode = 31 * functionName.hashCode() + argument.hashCode();
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
    return hashCode;
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
    return argument.equals(other.argument) && functionName.equals(other.functionName);
  }

  public static final Function<SkyKey, Object> NODE_NAME = new Function<SkyKey, Object>() {
    @Override
    public Object apply(SkyKey input) {
      return input.argument();
    }
  };
}
