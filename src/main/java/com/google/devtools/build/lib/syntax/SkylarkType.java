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
package com.google.devtools.build.lib.syntax;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.events.Location;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A class representing types available in Skylark.
 */
public class SkylarkType {

  public static final SkylarkType UNKNOWN = new SkylarkType(Object.class);
  public static final SkylarkType NONE = new SkylarkType(Environment.NoneType.class);

  public static final SkylarkType STRING = new SkylarkType(String.class);
  public static final SkylarkType INT = new SkylarkType(Integer.class);
  public static final SkylarkType BOOL = new SkylarkType(Boolean.class);

  private final Class<?> type;

  // TODO(bazel-team): Maybe change this to SkylarkType and allow list of lists etc.
  private Class<?> generic1;

  public static SkylarkType of(Class<?> type, Class<?> generic1) {
    return new SkylarkType(type, generic1);
  }

  public static SkylarkType of(Class<?> type) {
    return new SkylarkType(type);
  }

  private SkylarkType(Class<?> type, Class<?> generic1) {
    this.type = Preconditions.checkNotNull(type);
    this.generic1 = Preconditions.checkNotNull(generic1);
  }

  private SkylarkType(Class<?> type) {
    this.type = Preconditions.checkNotNull(type);
    this.generic1 = null;
  }

  Class<?> getType() {
    return type;
  }

  Class<?> getGenericType1() {
    return generic1;
  }

  /**
   * Returns the stronger type of this and o if they are compatible. Stronger means that
   * the more information is available, e.g. STRING is stronger than UNKNOWN and
   * LIST&lt;STRING> is stronger than LIST&lt;UNKNOWN>. Note than there's no type
   * hierarchy in Skylark.
   * <p>If they are not compatible an EvalException is thrown.
   */
  SkylarkType infer(SkylarkType o, String name, Location thisLoc, Location originalLoc)
      throws EvalException {
    if (this == o) {
      return this;
    }
    if (this == UNKNOWN) {
      return o;
    }
    if (o == UNKNOWN) {
      return this;
    }
    if (!type.equals(o.type)) {
      throw new EvalException(thisLoc, String.format("bad %s: %s is incompatible with %s at %s",
          name,
          EvalUtils.getDataTypeNameFromClass(o.getType()),
          EvalUtils.getDataTypeNameFromClass(this.getType()),
          originalLoc));
    }
    if (generic1 == null) {
      return o;
    }
    if (o.generic1 == null) {
      return this;
    }
    if (!generic1.equals(o.generic1)) {
      throw new EvalException(thisLoc, String.format("bad %s: incompatible generic variable types "
          + "%s with %s",
          name,
          EvalUtils.getDataTypeNameFromClass(o.generic1),
          EvalUtils.getDataTypeNameFromClass(this.generic1)));
    }
    return this;
  }

  boolean isStruct() {
    return type.equals(ClassObject.class);
  }

  boolean isList() {
    return List.class.isAssignableFrom(type);
  }

  boolean isDict() {
    return Map.class.isAssignableFrom(type);
  }

  boolean isSet() {
    return Set.class.isAssignableFrom(type);
  }

  boolean isNset() {
    // TODO(bazel-team): NestedSets are going to be a bit strange with 2 type info (validation
    // and execution time). That can be cleaned up once we have complete type inference.
    return SkylarkNestedSet.class.isAssignableFrom(type);
  }

  boolean isSimple() {
    return !isStruct() && !isDict() && !isList() && !isNset() && !isSet();
  }

  @Override
  public String toString() {
    return EvalUtils.getDataTypeNameFromClass(type);
  }
  // TODO(bazel-team): implement a special composite type for functions
}
