// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.errorprone.annotations.CanIgnoreReturnValue;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.SymbolGenerator;

/**
 * A context object, usually stored in a {@link StarlarkThread}, upon which rules and symbolic
 * macros can be instantiated.
 */
// TODO(#19922): Elevate some Package.Builder methods to this class.
public abstract class TargetDefinitionContext extends BazelStarlarkContext {

  /**
   * An exception used when the name of a target or symbolic macro clashes with another entity
   * defined in the package.
   *
   * <p>Common examples of conflicts include two targets or symbolic macros sharing the same name,
   * and one output file being a prefix of another. See {@link Package.Builder#checkForExistingName}
   * and {@link Package.Builder#checkRuleAndOutputs} for more details.
   */
  public static final class NameConflictException extends Exception {
    public NameConflictException(String message) {
      super(message);
    }
  }

  protected TargetDefinitionContext(Phase phase, SymbolGenerator<?> symbolGenerator) {
    super(phase, symbolGenerator);
  }

  /** Retrieves this object from a Starlark thread. Returns null if not present. */
  @Nullable
  public static TargetDefinitionContext fromOrNull(StarlarkThread thread) {
    BazelStarlarkContext ctx = thread.getThreadLocal(BazelStarlarkContext.class);
    return (ctx instanceof TargetDefinitionContext) ? (TargetDefinitionContext) ctx : null;
  }

  /**
   * Retrieves this object from a Starlark thread. If not present, throws {@code EvalException} with
   * an error message indicating that {@code what} can't be used in this Starlark environment.
   */
  @CanIgnoreReturnValue
  public static TargetDefinitionContext fromOrFail(StarlarkThread thread, String what)
      throws EvalException {
    @Nullable BazelStarlarkContext ctx = thread.getThreadLocal(BazelStarlarkContext.class);
    if (!(ctx instanceof TargetDefinitionContext)) {
      throw Starlark.errorf("%s can only be used while evaluating a BUILD file or macro", what);
    }
    return (TargetDefinitionContext) ctx;
  }
}
