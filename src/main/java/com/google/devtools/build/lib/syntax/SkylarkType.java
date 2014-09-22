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

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * A class representing types available in Skylark.
 */
public class SkylarkType {

  private static final class Global {}

  public static final SkylarkType UNKNOWN = new SkylarkType(Object.class);
  public static final SkylarkType NONE = new SkylarkType(Environment.NoneType.class);
  public static final SkylarkType GLOBAL = new SkylarkType(Global.class);

  public static final SkylarkType STRING = new SkylarkType(String.class);
  public static final SkylarkType INT = new SkylarkType(Integer.class);
  public static final SkylarkType BOOL = new SkylarkType(Boolean.class);

  private final Class<?> type;

  // TODO(bazel-team): Maybe change this to SkylarkType and allow list of lists etc.
  // Object.class is used for UNKNOWN.
  private Class<?> generic1;

  public static SkylarkType of(Class<?> type, Class<?> generic1) {
    return new SkylarkType(type, generic1);
  }

  public static SkylarkType of(Class<?> type) {
    if (type.equals(Object.class)) {
      return SkylarkType.UNKNOWN;
    } else if (type.equals(String.class)) {
      return SkylarkType.STRING;
    } else if (type.equals(Integer.class)) {
      return SkylarkType.INT;
    } else if (type.equals(Boolean.class)) {
      return SkylarkType.BOOL;
    }
    return new SkylarkType(type);
  }

  private SkylarkType(Class<?> type, Class<?> generic1) {
    this.type = Preconditions.checkNotNull(type);
    this.generic1 = Preconditions.checkNotNull(generic1);
  }

  private SkylarkType(Class<?> type) {
    this.type = Preconditions.checkNotNull(type);
    this.generic1 = Object.class;
  }

  public Class<?> getType() {
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
    if (generic1.equals(Object.class)) {
      return o;
    }
    if (o.generic1.equals(Object.class)) {
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

  boolean isCollection() {
    return Collection.class.isAssignableFrom(type);
  }

  @Override
  public String toString() {
    return this == UNKNOWN ? "Unknown" : EvalUtils.getDataTypeNameFromClass(type);
  }

  // hashCode() and equals() only uses the type field

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof SkylarkType)) {
      return false;
    }
    SkylarkType o = (SkylarkType) other;
    return this.type.equals(o.type);
  }

  @Override
  public int hashCode() {
    return type.hashCode();
  }

  /**
   * A class representing the type of a Skylark function.
   */
  public static final class SkylarkFunctionType extends SkylarkType {

    private final String name;
    @Nullable private SkylarkType returnType;
    @Nullable private Location returnTypeLoc;

    public static SkylarkFunctionType of(String name) {
      return new SkylarkFunctionType(name, null);
    }

    public static SkylarkFunctionType of(String name, SkylarkType returnType) {
      return new SkylarkFunctionType(name, returnType);
    }

    private SkylarkFunctionType(String name, SkylarkType returnType) {
      super(Function.class);
      this.name = name;
      this.returnType = returnType;
    }

    public SkylarkType getReturnType() {
      return returnType;
    }

    /**
     * Sets the return type of the function type if it's compatible with the existing return type.
     * Note that setting NONE only has an effect if the return type hasn't been set previously.
     */
    public void setReturnType(SkylarkType newReturnType, Location newLoc) throws EvalException {
      if (returnType == null) {
        returnType = newReturnType;
        returnTypeLoc = newLoc;
      } else if (newReturnType != SkylarkType.NONE) {
        returnType =
            returnType.infer(newReturnType, "return type of " + name, newLoc, returnTypeLoc);
        if (returnType == newReturnType) {
          returnTypeLoc = newLoc;
        }
      }
    }
  }
}
