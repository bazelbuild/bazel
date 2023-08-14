// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.Structure;

/**
 * A helper class for calling Starlark functions from Java, where the argument values are supplied
 * by the fields of a Structure, as in the case of computed attribute defaults and computed implicit
 * outputs.
 */
// TODO(brandjon): Consider eliminating this class by executing the callback in the same thread as
// the caller, i.e. in the thread evaluating a BUILD file. This might not be possible for implicit
// outputs without some refactoring to cache the result of the computation (currently RuleContext
// seems to reinvoke the callback).
public final class StarlarkCallbackHelper {

  private final StarlarkFunction callback;

  // These fields, parts of the state of the loading-phase
  // thread that instantiated a rule, must be propagated to
  // the child threads (implicit outputs, attribute defaults).
  // This includes any other thread-local state, such as
  // PackageFactory.PackageContext.
  // TODO(adonovan): it would be cleaner and less error prone to
  // perform these callbacks in the actual loading-phase thread,
  // at the end of BUILD file execution.
  // Alternatively (or additionally), we could put PackageContext
  // into BazelStarlarkContext so there's only a single blob of state.
  private final StarlarkSemantics starlarkSemantics;

  public StarlarkCallbackHelper(StarlarkFunction callback, StarlarkSemantics starlarkSemantics) {
    this.callback = callback;
    this.starlarkSemantics = starlarkSemantics;
  }

  public ImmutableList<String> getParameterNames() {
    return callback.getParameterNames();
  }

  // TODO(adonovan): opt: all current callers are forced to construct a temporary Structure.
  // Instead, make them supply a map.
  public Object call(EventHandler eventHandler, Structure struct, Object... arguments)
      throws EvalException, InterruptedException {
    try (Mutability mu = Mutability.create("callback", callback)) {
      StarlarkThread thread = new StarlarkThread(mu, starlarkSemantics);
      thread.setPrintHandler(Event.makeDebugPrintHandler(eventHandler));
      new BazelStarlarkContext(
              BazelStarlarkContext.Phase.LOADING,
              // TODO(brandjon): In principle, if we're creating a new symbol generator here, we
              // should have a unique owner object to associate it with for distinguishing
              // reference-equality objects. But I don't think implicit outputs or computed defaults
              // care about identity.
              new SymbolGenerator<>(new Object()),
              /* analysisRuleLabel= */ null)
          .storeInThread(thread);
      return Starlark.call(
          thread, callback, buildArgumentList(struct, arguments), /*kwargs=*/ ImmutableMap.of());
    } catch (ClassCastException | IllegalArgumentException e) { // TODO(adonovan): investigate
      throw new EvalException(e);
    }
  }

  /**
   * Creates a list of actual arguments that contains the given arguments and all attribute values
   * required from the specified structure.
   */
  private ImmutableList<Object> buildArgumentList(Structure struct, Object... arguments)
      throws EvalException {
    ImmutableList.Builder<Object> builder = ImmutableList.builder();
    ImmutableList<String> names = getParameterNames();
    int requiredParameters = names.size() - arguments.length;
    for (int pos = 0; pos < requiredParameters; ++pos) {
      String name = names.get(pos);
      Object value = struct.getValue(name);
      if (value == null) {
        throw new IllegalArgumentException(struct.getErrorMessageForUnknownField(name));
      }
      builder.add(value);
    }
    return builder.add(arguments).build();
  }
}
