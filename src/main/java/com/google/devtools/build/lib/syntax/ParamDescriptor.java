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
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;
import java.util.Arrays;
import javax.annotation.Nullable;

/** A value class for storing {@link Param} metadata to avoid using Java proxies. */
final class ParamDescriptor {

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

  // The next two fields relate to toggling this parameter via semantic flag -- they will
  // be null if and only if this parameter is enabled, and will otherwise contain information
  // about what to do with the disabled parameter. (If the parameter is 'disabled', it will be
  // treated as unusable from Starlark.)

  // The value of this disabled parameter (as interpreted in Starlark) will be passed to the Java
  // method.
  @Nullable private final String valueOverride;
  // The flag responsible for disabling this parameter. If a user attempts to use this disabled
  // parameter from Starlark, this identifier can be used to create the appropriate error message.
  @Nullable private final FlagIdentifier flagResponsibleForDisable;

  private ParamDescriptor(
      String name,
      String defaultValue,
      Class<?> type,
      ImmutableList<ParamTypeDescriptor> allowedTypes,
      Class<?> generic1,
      boolean noneable,
      boolean named,
      boolean positional,
      SkylarkType skylarkType,
      @Nullable String valueOverride,
      @Nullable FlagIdentifier flagResponsibleForDisable) {
    this.name = name;
    this.defaultValue = defaultValue;
    this.type = type;
    this.allowedTypes = allowedTypes;
    this.generic1 = generic1;
    this.noneable = noneable;
    this.named = named;
    this.positional = positional;
    this.skylarkType = skylarkType;
    this.valueOverride = valueOverride;
    this.flagResponsibleForDisable = flagResponsibleForDisable;
  }

  /**
   * Returns a {@link ParamDescriptor} representing the given raw {@link Param} annotation and the
   * given semantics.
   */
  static ParamDescriptor of(Param param, StarlarkSemantics starlarkSemantics) {
    ImmutableList<ParamTypeDescriptor> allowedTypes =
        Arrays.stream(param.allowedTypes())
            .map(ParamTypeDescriptor::of)
            .collect(ImmutableList.toImmutableList());
    Class<?> type = param.type();
    Class<?> generic = param.generic1();
    boolean noneable = param.noneable();

    boolean isParamEnabledWithCurrentSemantics =
        starlarkSemantics.isFeatureEnabledBasedOnTogglingFlags(
            param.enableOnlyWithFlag(), param.disableWithFlag());

    String valueOverride = null;
    FlagIdentifier flagResponsibleForDisable = FlagIdentifier.NONE;
    if (!isParamEnabledWithCurrentSemantics) {
      valueOverride = param.valueWhenDisabled();
      flagResponsibleForDisable =
          param.enableOnlyWithFlag() != FlagIdentifier.NONE
              ? param.enableOnlyWithFlag()
              : param.disableWithFlag();
    }
    return new ParamDescriptor(
        param.name(),
        param.defaultValue(),
        type,
        allowedTypes,
        generic,
        noneable,
        param.named()
            || (param.legacyNamed() && !starlarkSemantics.incompatibleRestrictNamedParams()),
        param.positional(),
        getType(type, generic, allowedTypes, noneable),
        valueOverride,
        flagResponsibleForDisable);
  }

  /** @see Param#name() */
  String getName() {
    return name;
  }

  /** @see Param#allowedTypes() */
  ImmutableList<ParamTypeDescriptor> getAllowedTypes() {
    return allowedTypes;
  }

  /** @see Param#type() */
  Class<?> getType() {
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
  Class<?> getGeneric1() {
    return generic1;
  }

  /** @see Param#noneable() */
  boolean isNoneable() {
    return noneable;
  }

  /** @see Param#positional() */
  boolean isPositional() {
    return positional;
  }

  /** @see Param#named() */
  boolean isNamed() {
    return named;
  }

  /** @see Param#defaultValue() */
  String getDefaultValue() {
    return defaultValue;
  }

  SkylarkType getSkylarkType() {
    return skylarkType;
  }

  /** Returns true if this parameter is disabled under the current skylark semantic flags. */
  boolean isDisabledInCurrentSemantics() {
    return valueOverride != null;
  }

  /**
   * Returns the value the parameter should take, given that the parameter is disabled under the
   * current skylark semantics.
   *
   * @throws IllegalStateException if invoked when {@link #isDisabledInCurrentSemantics()} is false
   */
  String getValueOverride() {
    Preconditions.checkState(
        isDisabledInCurrentSemantics(),
        "parameter is not disabled under the current semantic flags. getValueOverride should be "
            + "called only if isParameterDisabled is true");
    return valueOverride;
  }

  /**
   * Returns the flag responsible for disabling this parameter, given that the parameter is disabled
   * under the current skylark semantics.
   *
   * @throws IllegalStateException if invoked when {@link #isDisabledInCurrentSemantics()} is false
   */
  FlagIdentifier getFlagResponsibleForDisable() {
    Preconditions.checkState(
        isDisabledInCurrentSemantics(),
        "parameter is not disabled under the current semantic flags. getFlagResponsibleForDisable "
            + " should be called only if isParameterDisabled is true");
    return flagResponsibleForDisable;
  }
}
