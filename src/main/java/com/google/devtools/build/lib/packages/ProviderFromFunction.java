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
package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.SkylarkType;
import javax.annotation.Nullable;

/**
 * Implementation of {@link Provider} that is represented by a {@link BaseFunction}, and thus
 * callable from Skylark.
 */
@Immutable
abstract class ProviderFromFunction extends BaseFunction implements Provider {

  /**
   * Constructs a provider.
   *
   * @param name provider name; should be null iff the subclass overrides {@link #getName}
   * @param signature the signature for calling this provider as a Skylark function (to construct an
   *     instance of the provider)
   * @param location the location of this provider's Skylark definition. Use {@link
   *     Location#BUILTIN} if it is a native provider.
   */
  protected ProviderFromFunction(
      @Nullable String name,
      FunctionSignature.WithValues<Object, SkylarkType> signature,
      Location location) {
    super(name, signature, location);
  }

  public SkylarkProviderIdentifier id() {
    return SkylarkProviderIdentifier.forKey(getKey());
  }

  @Override
  protected Object call(Object[] args, @Nullable FuncallExpression ast, Environment env)
      throws EvalException, InterruptedException {
    Location loc = ast != null ? ast.getLocation() : Location.BUILTIN;
    return createInstanceFromSkylark(args, env, loc);
  }

  /**
   * Override this method to provide logic that is used to instantiate a declared provider from
   * Skylark.
   *
   * <p>This is a method that is called when a constructor {@code c} is invoked as<br>
   * {@code c(arg1 = val1, arg2 = val2, ...)}.
   *
   * @param args an array of argument values sorted as per the signature ({@see BaseFunction#call})
   */
  protected abstract InfoInterface createInstanceFromSkylark(
      Object[] args, Environment env, Location loc) throws EvalException;
}
