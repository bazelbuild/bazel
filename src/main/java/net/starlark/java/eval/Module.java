// Copyright 2019 The Bazel Authors. All rights reserved.
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

package net.starlark.java.eval;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Resolver;

/**
 * A {@link Module} represents a Starlark module, a container of global variables populated by
 * executing a Starlark file. Each top-level assignment updates a global variable in the module.
 *
 * <p>Each module references its "predeclared" environment, which is often shared among many
 * modules. These are the names that are defined even at the start of execution. For example, in
 * Bazel, the predeclared environment of the module for a BUILD or .bzl file defines name values
 * such as cc_binary and glob.
 *
 * <p>The predeclared environment implicitly includes the "universal" names present in every
 * Starlark thread in every dialect, such as None, len, and str; see {@link Starlark#UNIVERSE}.
 *
 * <p>Global bindings in a Module may shadow bindings inherited from the predeclared block.
 *
 * <p>A module may carry an arbitrary piece of client data. In Bazel, for example, the client data
 * records the module's build label (such as "//dir:file.bzl"). This client data is accessible to
 * (for instance) application-defined builtin methods.
 *
 * <p>You may create a Module using {@link #create}, {@link #withPredeclared}, or {@link
 * #withPredeclaredAndData}. The latter two give you the ability to add predeclared bindings (beyond
 * the universal ones) and client data. The particular {@link StarlarkSemantics} and client data may
 * filter what predeclared bindings are available via {@link GuardedValue}.
 */
public final class Module implements Resolver.Module {

  // The module's predeclared environment. Excludes UNIVERSE bindings. Values that are conditionally
  // present are stored as GuardedValues regardless of whether they are actually enabled.
  private final ImmutableMap<String, Object> predeclared;

  // The module's global variables, in order of creation.
  private final LinkedHashMap<String, Integer> globalIndex = new LinkedHashMap<>();
  private Object[] globals = new Object[8];

  // An optional piece of application-specific metadata associated with the module/file.
  // Its toString appears to Starlark in str(function): "<function f from ...>".
  @Nullable private final Object clientData;

  private final StarlarkSemantics semantics;

  // An optional doc string for the module. Set after construction when evaluating a .bzl file.
  @Nullable private String documentation;

  private Module(
      ImmutableMap<String, Object> predeclared,
      @Nullable Object clientData,
      StarlarkSemantics semantics) {
    this.predeclared = predeclared;
    this.clientData = clientData;
    this.semantics = semantics;
  }

  /**
   * Constructs a Module with the specified predeclared bindings (filtered by the semantics), in *
   * addition to the standard environment, {@link Starlark#UNIVERSE}. No client data is set.
   */
  public static Module withPredeclared(
      StarlarkSemantics semantics, Map<String, Object> predeclared) {
    return withPredeclaredAndData(semantics, predeclared, null);
  }

  /**
   * Constructs a Module as above, but with the specified client data -- an arbitrary
   * application-specific value to be associated with this Module. Client data may also affect the
   * filtering of predeclareds alongside the semantics.
   */
  public static Module withPredeclaredAndData(
      StarlarkSemantics semantics, Map<String, Object> predeclared, @Nullable Object clientData) {
    return new Module(ImmutableMap.copyOf(predeclared), clientData, semantics);
  }

  /**
   * Creates a module with no predeclared bindings other than the standard environment, {@link
   * Starlark#UNIVERSE}, and with no client data.
   */
  public static Module create() {
    return new Module(
        /* predeclared= */ ImmutableMap.of(), /* clientData= */ null, StarlarkSemantics.DEFAULT);
  }

  /**
   * Returns the module (file) of the {@code depth}-th innermost enclosing Starlark function on the
   * call stack, or null if number of the active calls that are functions defined in Starlark is
   * less than or equal to {@code depth}.
   *
   * <p>This method is a temporary workaround for Starlarkification, to check {@code _builtin}
   * restriction and should not be used anywhere else.
   *
   * @param depth the depth for the callstack.
   * @throws IllegalArgumentException if {@code depth} is negative.
   */
  @Nullable
  public static Module ofInnermostEnclosingStarlarkFunction(StarlarkThread thread, int depth) {
    StarlarkFunction fn = thread.getInnermostEnclosingStarlarkFunction(depth);
    if (fn != null) {
      return fn.getModule();
    }
    return null;
  }

  /**
   * Returns the module (file) of the innermost enclosing Starlark function on the call stack, or
   * null if none of the active calls are functions defined in Starlark.
   *
   * <p>The name of this function is intentionally horrible to make you feel bad for using it.
   */
  @Nullable
  public static Module ofInnermostEnclosingStarlarkFunction(StarlarkThread thread) {
    return ofInnermostEnclosingStarlarkFunction(thread, 0);
  }

  /**
   * Replaces an enabled {@link GuardedValue} with the value it guards.
   *
   * <p>A disabled {@link GuardedValue} is left in place for error reporting upon access, and should
   * be treated as unavailable.
   */
  private Object filterGuardedValue(Object v) {
    Preconditions.checkNotNull(v);
    if (!(v instanceof GuardedValue)) {
      return v;
    }
    GuardedValue gv = (GuardedValue) v;
    return gv.isObjectAccessibleUsingSemantics(semantics, clientData) ? gv.getObject() : gv;
  }

  /** Returns the client data associated with this module. */
  @Nullable
  public Object getClientData() {
    return clientData;
  }

  /** Sets the module's doc string. It may be retrieved using {@link #getDocumentation}. */
  public void setDocumentation(String documentation) {
    this.documentation = documentation;
  }

  /**
   * Returns the module's doc string, or null if absent.
   *
   * <p>Morally equivalent to calling {@code program.getResolvedFunction().getDocumentation()} when
   * the Module has a corresponding {@link net.starlark.java.syntax.Program}. We need to separately
   * save the doc string inside the Module because (1) a Module will usually outlive the Program;
   * and (2) there isn't always a 1-to-1 match between a Module and a Program (multiple programs may
   * be executed in the same module in REPL or in tests).
   */
  @Nullable
  public String getDocumentation() {
    return documentation;
  }

  /**
   * Returns the value of a predeclared (not universal) binding in this module.
   *
   * <p>In the case that the predeclared is a {@link GuardedValue}: If it is enabled, the underlying
   * value is returned, otherwise the {@code GuardedValue} itself is returned for error reporting.
   */
  @Nullable
  public Object getPredeclared(String name) {
    var value = predeclared.get(name);
    if (value == null) {
      return null;
    }
    return filterGuardedValue(value);
  }

  /**
   * Returns this module's additional predeclared bindings. (Excludes {@link Starlark#UNIVERSE}.)
   *
   * <p>The map reflects any filtering of {@link GuardedValue}: enabled ones are replaced by the
   * underlying values that they guard, while disabled ones are left in place for error reporting.
   */
  public Map<String, Object> getPredeclaredBindings() {
    return Maps.transformValues(predeclared, this::filterGuardedValue);
  }

  /**
   * Returns an immutable mapping containing the global variables of this module.
   *
   * <p>The bindings are returned in a deterministic order (for a given sequence of initial values
   * and updates).
   */
  public ImmutableMap<String, Object> getGlobals() {
    int n = globalIndex.size();
    ImmutableMap.Builder<String, Object> m = ImmutableMap.builderWithExpectedSize(n);
    for (Map.Entry<String, Integer> e : globalIndex.entrySet()) {
      Object v = getGlobalByIndex(e.getValue());
      if (v != null) {
        m.put(e.getKey(), v);
      }
    }
    return m.buildOrThrow();
  }

  /** Implements the resolver's module interface. */
  @Override
  public Resolver.Scope resolve(String name) throws Undefined {
    // global?
    if (globalIndex.containsKey(name)) {
      return Resolver.Scope.GLOBAL;
    }

    // predeclared?
    Object v = getPredeclared(name);
    if (v != null) {
      if (v instanceof GuardedValue) {
        // Name is correctly spelled, but access is disabled by a flag or by client data.
        throw new Undefined(
            ((GuardedValue) v).getErrorFromAttemptingAccess(name), /*candidates=*/ null);
      }
      return Resolver.Scope.PREDECLARED;
    }

    // universal?
    if (Starlark.UNIVERSE.containsKey(name)) {
      return Resolver.Scope.UNIVERSAL;
    }

    // undefined
    Set<String> candidates = new HashSet<>();
    candidates.addAll(globalIndex.keySet());
    candidates.addAll(predeclared.keySet());
    candidates.addAll(Starlark.UNIVERSE.keySet());
    throw new Undefined(String.format("name '%s' is not defined", name), candidates);
  }

  /**
   * Returns the value of the specified global variable, or null if not bound. Does not look in the
   * predeclared environment.
   */
  @Nullable
  public Object getGlobal(String name) {
    Integer i = globalIndex.get(name);
    return i != null ? globals[i] : null;
  }

  /**
   * Sets the value of a global variable based on its index in this module ({@see
   * getIndexOfGlobal}).
   */
  void setGlobalByIndex(int i, Object v) {
    Preconditions.checkArgument(i < globalIndex.size());
    this.globals[i] = v;
  }

  /**
   * Returns the value of a global variable based on its index in this module ({@see
   * getIndexOfGlobal}.) Returns null if the variable has not been assigned a value.
   */
  @Nullable
  Object getGlobalByIndex(int i) {
    Preconditions.checkArgument(i < globalIndex.size());
    return this.globals[i];
  }

  /**
   * Returns the index within this Module of a global variable, given its name, creating a new slot
   * for it if needed. The numbering of globals used by these functions is not the same as the
   * numbering within any compiled Program. Thus each StarlarkFunction must contain a secondary
   * index mapping Program indices (from Binding.index) to Module indices.
   */
  int getIndexOfGlobal(String name) {
    int i = globalIndex.size();
    Integer prev = globalIndex.putIfAbsent(name, i);
    if (prev != null) {
      return prev;
    }
    if (i == globals.length) {
      globals = Arrays.copyOf(globals, globals.length << 1); // grow by doubling
    }
    return i;
  }

  /** Returns a list of indices of a list of globals; {@see getIndexOfGlobal}. */
  int[] getIndicesOfGlobals(List<String> globals) {
    int n = globals.size();
    int[] array = new int[n];
    for (int i = 0; i < n; i++) {
      array[i] = getIndexOfGlobal(globals.get(i));
    }
    return array;
  }

  /** Updates a global binding in the module environment. */
  public void setGlobal(String name, Object value) {
    Preconditions.checkNotNull(value, "Module.setGlobal(%s, null)", name);
    setGlobalByIndex(getIndexOfGlobal(name), value);
  }

  @Override
  public String toString() {
    return String.format("<module %s>", clientData == null ? "?" : clientData);
  }
}
