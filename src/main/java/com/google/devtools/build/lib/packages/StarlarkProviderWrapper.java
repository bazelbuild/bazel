// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;

/**
 * A helper for wrapping an instance of a Starlark-defined provider with a native class {@code T}.
 *
 * <p>This is useful for allowing native rules to interoperate with Starlark-defined providers
 * (including providers defined in {@code @_builtins}} while retaining the friendlier API of a Java
 * class.
 *
 * <p>To use, create a subclass that overrides {@link #wrap}, and pass a singleton instance of that
 * subclass to {@link
 * com.google.devtools.build.lib.analysis.ProviderCollection#get(StarlarkProviderWrapper)}.
 *
 * <p>{@code T} is not typically itself a provider. There is no mechanism for converting {@code T}
 * back into a Starlark provider instance; instead, the caller should construct that instance
 * manually.
 */
@Immutable
public abstract class StarlarkProviderWrapper<T> {
  private final StarlarkProvider.Key key;

  protected StarlarkProviderWrapper(BzlLoadValue.Key loadKey, String name) {
    this.key = new StarlarkProvider.Key(loadKey, name);
  }

  /**
   * Converts an instance of the Starlark-defined provider to an instance of the wrapping class
   * {@code T}.
   *
   * <p>{@code value} may be assumed to be an instance of the provider identified by {@link
   * #getKey}.
   *
   * <p>Any schema errors (missing or mistyped fields) should be reported by throwing {@link
   * RuleErrorException}
   */
  public abstract T wrap(Info value) throws RuleErrorException;

  public StarlarkProvider.Key getKey() {
    return key;
  }

  /** Returns the identifier of this provider. */
  public StarlarkProviderIdentifier id() {
    return StarlarkProviderIdentifier.forKey(key);
  }
}
