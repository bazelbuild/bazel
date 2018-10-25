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

package com.google.devtools.build.lib.syntax;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skylarkinterface.Param;
import java.util.Arrays;

/** A value class for storing {@link Param} metadata to avoid using Java proxies. */
public final class ParamDescriptor {

  private final String name;
  private final String defaultValue;
  private final Class<?> type;
  private final ImmutableList<ParamTypeDescriptor> allowedTypes;
  private final Class<?> generic1;
  private final boolean noneable;
  private final boolean named;
  private final boolean positional;
  // While the type can be inferred completely by the Param annotation, this tuple allows for the
  // type of a given parameter to be determined only once, as it is an expensive operation.
  private final SkylarkType skylarkType;

  private ParamDescriptor(
      String name,
      String defaultValue,
      Class<?> type,
      ImmutableList<ParamTypeDescriptor> allowedTypes,
      Class<?> generic1,
      boolean noneable,
      boolean named,
      boolean legacyNamed,
      boolean positional,
      SkylarkType skylarkType) {
    this.name = name;
    this.defaultValue = defaultValue;
    this.type = type;
    this.allowedTypes = allowedTypes;
    this.generic1 = generic1;
    this.noneable = noneable;
    this.named = named || legacyNamed;
    this.positional = positional;
    this.skylarkType = skylarkType;
  }

  static ParamDescriptor of(Param param) {
    ImmutableList<ParamTypeDescriptor> allowedTypes =
        Arrays.stream(param.allowedTypes())
            .map(ParamTypeDescriptor::of)
            .collect(ImmutableList.toImmutableList());
    Class<?> type = param.type();
    Class<?> generic = param.generic1();
    boolean noneable = param.noneable();
    return new ParamDescriptor(
        param.name(),
        param.defaultValue(),
        type,
        allowedTypes,
        generic,
        noneable,
        param.named(),
        param.legacyNamed(),
        param.positional(),
        getType(type, generic, allowedTypes, noneable));
  }

  /** @see Param#name() */
  public String getName() {
    return name;
  }

  /** @see Param#allowedTypes() */
  public ImmutableList<ParamTypeDescriptor> getAllowedTypes() {
    return allowedTypes;
  }

  /** @see Param#type() */
  public Class<?> getType() {
    return type;
  }

  private static SkylarkType getType(
      Class<?> type,
      Class<?> generic,
      ImmutableList<ParamTypeDescriptor> allowedTypes,
      boolean noneable) {
    SkylarkType result = SkylarkType.BOTTOM;
    if (!allowedTypes.isEmpty()) {
      Preconditions.checkState(Object.class.equals(type));
      for (ParamTypeDescriptor paramType : allowedTypes) {
        SkylarkType t =
            paramType.getGeneric1() != Object.class
                ? SkylarkType.of(paramType.getType(), paramType.getGeneric1())
                : SkylarkType.of(paramType.getType());
        result = SkylarkType.Union.of(result, t);
      }
    } else {
      result = generic != Object.class ? SkylarkType.of(type, generic) : SkylarkType.of(type);
    }

    if (noneable) {
      result = SkylarkType.Union.of(result, SkylarkType.NONE);
    }
    return result;
  }

  /** @see Param#generic1() */
  public Class<?> getGeneric1() {
    return generic1;
  }

  /** @see Param#noneable() */
  public boolean isNoneable() {
    return noneable;
  }

  /** @see Param#positional() */
  public boolean isPositional() {
    return positional;
  }

  /** @see Param#named() */
  public boolean isNamed() {
    return named;
  }

  /** @see Param#defaultValue() */
  public String getDefaultValue() {
    return defaultValue;
  }

  SkylarkType getSkylarkType() {
    return skylarkType;
  }
}
