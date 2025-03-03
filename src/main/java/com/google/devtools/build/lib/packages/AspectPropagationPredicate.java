// Copyright 2025 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.cmdline.StarlarkThreadContext;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkAspectPropagationContextApi;
import java.util.Objects;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;

/** Starlark function that determines whether aspect should be propagated to the target. */
public final class AspectPropagationPredicate {

  private final StarlarkFunction predicate;
  private final StarlarkSemantics semantics;

  public AspectPropagationPredicate(StarlarkFunction predicate, StarlarkSemantics semantics) {
    this.predicate = predicate;
    this.semantics = semantics;
  }

  public boolean evaluate(
      StarlarkAspectPropagationContextApi context, ExtendedEventHandler eventHandler)
      throws InterruptedException, EvalException {

    Object starlarkResult = runPropagationPredicate(context, eventHandler);

    if (starlarkResult instanceof Boolean booleanResult) {
      return booleanResult;
    }
    throw new EvalException("Expected a boolean");
  }

  private Object runPropagationPredicate(
      StarlarkAspectPropagationContextApi context, ExtendedEventHandler eventHandler)
      throws InterruptedException, EvalException {
    try (Mutability mu = Mutability.create("aspect_propagation_predicate")) {
      StarlarkThread thread = StarlarkThread.createTransient(mu, semantics);
      thread.setPrintHandler(Event.makeDebugPrintHandler(eventHandler));

      new AspectPropagationThreadContext().storeInThread(thread);
      return Starlark.positionalOnlyCall(thread, predicate, context);
    }
  }

  private static class AspectPropagationThreadContext extends StarlarkThreadContext {
    public AspectPropagationThreadContext() {
      super(null);
    }
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    AspectPropagationPredicate that = (AspectPropagationPredicate) o;
    return Objects.equals(predicate, that.predicate) && Objects.equals(semantics, that.semantics);
  }

  @Override
  public int hashCode() {
    return Objects.hash(predicate, semantics);
  }
}
