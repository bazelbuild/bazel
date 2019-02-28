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
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;

/**
 * Wrapper on a value that controls its accessibility in Starlark based on the value of a
 * semantic flag.
 *
 * <p>For example, this could control whether symbol "Foo" exists in the Starlark
 * global frame: such a symbol might only be accessible if --experimental_foo is set to true.
 * In order to create this control, an instance of this class should be added to the global
 * frame under "Foo". This flag guard will throw a descriptive {@link EvalException} when
 * "Foo" would be accessed without the proper flag.
 */
public class FlagGuardedValue {
  private final Object obj;
  private final FlagIdentifier flagIdentifier;
  private final FlagType flagType;

  private enum FlagType {
    DEPRECATION,
    EXPERIMENTAL;
  }

  private FlagGuardedValue(Object obj, FlagIdentifier flagIdentifier, FlagType flagType) {
    this.obj = obj;
    this.flagIdentifier = flagIdentifier;
    this.flagType = flagType;
  }

  /**
   * Creates a flag guard which only permits access of the given object when the given flag is
   * true. If the given flag is false and the object would be accessed, an error is thrown
   * describing the feature as experimental, and describing that the flag must be set to true.
   */
  public static FlagGuardedValue onlyWhenExperimentalFlagIsTrue(
      FlagIdentifier flagIdentifier, Object obj) {
    return new FlagGuardedValue(obj, flagIdentifier, FlagType.EXPERIMENTAL);
  }

  /**
   * Creates a flag guard which only permits access of the given object when the given flag is
   * false. If the given flag is true and the object would be accessed, an error is thrown
   * describing the feature as deprecated, and describing that the flag must be set to false.
   */
  public static FlagGuardedValue onlyWhenIncompatibleFlagIsFalse(
      FlagIdentifier flagIdentifier, Object obj) {
    return new FlagGuardedValue(obj, flagIdentifier, FlagType.DEPRECATION);
  }

  /**
   * Returns an {@link EvalException} with error appropriate to throw when one attempts to access
   * this guard's protected object when it should be inaccessible in the given semantics.
   *
   * @throws IllegalArgumentException if {@link #isObjectAccessibleUsingSemantics} is true given the
   *     semantics
   */
  public EvalException getEvalExceptionFromAttemptingAccess(
      Location location, StarlarkSemantics semantics, String symbolDescription) {
    Preconditions.checkArgument(!isObjectAccessibleUsingSemantics(semantics),
        "getEvalExceptionFromAttemptingAccess should only be called if the underlying "
            + "object is inaccessible given the semantics");
    if (flagType == FlagType.EXPERIMENTAL) {
      return new EvalException(
            location,
            symbolDescription
                + " is experimental and thus unavailable with the current flags. It may be "
                + "enabled by setting --" + flagIdentifier.getFlagName());
    } else {
      return new EvalException(
        location,
        symbolDescription
            + " is deprecated and will be removed soon. It may be temporarily re-enabled by "
            + "setting --" + flagIdentifier.getFlagName() + "=false");

    }
  }

  /**
   * Returns this guard's underlying object. This should be called when appropriate validation has
   * occurred to ensure that the object is accessible with the given semantics.
   *
   * @throws IllegalArgumentException if {@link #isObjectAccessibleUsingSemantics} is false given
   *     the semantics
   */
  public Object getObject(StarlarkSemantics semantics) {
    Preconditions.checkArgument(isObjectAccessibleUsingSemantics(semantics),
        "getObject should only be called if the underlying object is accessible given the "
            + "semantics");
    return obj;
  }

  /** Returns true if this guard's underlying object is accessible under the given semantics. */
  public boolean isObjectAccessibleUsingSemantics(StarlarkSemantics semantics) {
    if (flagType == FlagType.EXPERIMENTAL) {
      return semantics.isFeatureEnabledBasedOnTogglingFlags(flagIdentifier, FlagIdentifier.NONE);
    } else {
      return semantics.isFeatureEnabledBasedOnTogglingFlags(FlagIdentifier.NONE, flagIdentifier);
    }
  }
}
