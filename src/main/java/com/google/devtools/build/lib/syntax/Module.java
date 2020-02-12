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

package com.google.devtools.build.lib.syntax;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.syntax.Mutability.MutabilityException;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * A {@link Module} represents a Starlark module, a container of global variables populated by
 * executing a Starlark file. Each top-level assignment updates a global variable in the module.
 *
 * <p>Each module references its "predeclared" environment, which is often shared among many
 * modules. These are the names that are defined even at the start of execution. For example, in
 * Bazel, the predeclared environment of the module for a BUILD or .bzl file defines name values
 * such as cc_binary and glob.
 *
 * <p>The predeclared environment currently must include the "universal" names present in every
 * Starlark thread in every dialect, such as None, len, and str.
 *
 * <p>Global bindings in a Module may shadow bindings inherited from the predeclared or universe
 * block.
 *
 * <p>A module may carry an arbitrary piece of metadata called its "label". In Bazel, for example,
 * the label is a build label such as "//dir:file.bzl", for use by the Label function. This is a
 * hack.
 *
 * <p>A {@link Module} may be constructed in a two-phase process. To do this, call the nullary
 * constructor to create an uninitialized {@link Module}, then call {@link #initialize}. It is
 * illegal to use any other method in-between these two calls, or to call {@link #initialize} on an
 * already initialized {@link Module}.
 */
// TODO(adonovan):
// - make fields private where possible.
// - remove references to this from StarlarkThread.
// - separate the universal predeclared environment and make it implicit.
// - eliminate initialize(). The only constructor we need is:
//   (String name, Mutability mu, Map<String, Object> predeclared, Object label).
public final class Module implements ValidationEnvironment.Module {

  /**
   * Final, except that it may be initialized after instantiation. Null mutability indicates that
   * this Frame is uninitialized.
   */
  @Nullable private Mutability mutability;

  /** Final, except that it may be initialized after instantiation. */
  @Nullable Module universe;

  // The label (an optional piece of metadata) associated with the file.
  @Nullable Object label;

  /** Bindings are maintained in order of creation. */
  private final LinkedHashMap<String, Object> bindings;

  /**
   * A list of bindings which *would* exist in this global frame under certain semantic flags, but
   * do not exist using the semantic flags used in this frame's creation. This map should not be
   * used for lookups; it should only be used to throw descriptive error messages when a lookup of a
   * restricted object is attempted.
   */
  final LinkedHashMap<String, FlagGuardedValue> restrictedBindings;

  /** Set of bindings that are exported (can be loaded from other modules). */
  final HashSet<String> exportedBindings;

  /** Constructs an uninitialized instance; caller must call {@link #initialize} before use. */
  public Module() {
    this.mutability = null;
    this.universe = null;
    this.label = null;
    this.bindings = new LinkedHashMap<>();
    this.restrictedBindings = new LinkedHashMap<>();
    this.exportedBindings = new HashSet<>();
  }

  /**
   * Returns the module (file) of the innermost enclosing Starlark function on the call stack, or
   * null if none of the active calls are functions defined in Starlark.
   *
   * <p>The name of this function is intentionally horrible to make you feel bad for using it.
   */
  @Nullable
  public static Module ofInnermostEnclosingStarlarkFunction(StarlarkThread thread) {
    for (Debug.Frame fr : thread.getDebugCallStack().reverse()) {
      if (fr.getFunction() instanceof StarlarkFunction) {
        return ((StarlarkFunction) fr.getFunction()).getModule();
      }
    }
    return null;
  }

  Module(
      Mutability mutability,
      @Nullable Module universe,
      @Nullable Object label,
      @Nullable Map<String, Object> bindings,
      @Nullable Map<String, FlagGuardedValue> restrictedBindings) {
    Preconditions.checkState(universe == null || universe.universe == null);
    this.mutability = Preconditions.checkNotNull(mutability);
    this.universe = universe;
    if (label != null) {
      this.label = label;
    } else if (universe != null) {
      this.label = universe.label;
    } else {
      this.label = null;
    }
    this.bindings = new LinkedHashMap<>();
    if (bindings != null) {
      this.bindings.putAll(bindings);
    }
    this.restrictedBindings = new LinkedHashMap<>();
    if (restrictedBindings != null) {
      this.restrictedBindings.putAll(restrictedBindings);
    }
    if (universe != null) {
      this.restrictedBindings.putAll(universe.restrictedBindings);
    }
    this.exportedBindings = new HashSet<>();
  }

  public Module(Mutability mutability) {
    this(mutability, null, null, null, null);
  }

  public Module(Mutability mutability, @Nullable Module universe) {
    this(mutability, universe, null, null, null);
  }

  public Module(Mutability mutability, @Nullable Module universe, @Nullable Object label) {
    this(mutability, universe, label, null, null);
  }

  /** Constructs a global frame for the given builtin bindings. */
  public static Module createForBuiltins(Map<String, Object> bindings) {
    Mutability mutability = Mutability.create("<builtins>").freeze();
    return new Module(mutability, null, null, bindings, null);
  }

  /**
   * Constructs a global frame based on the given parent frame, filtering out flag-restricted global
   * objects.
   */
  static Module filterOutRestrictedBindings(
      Mutability mutability, Module parent, StarlarkSemantics semantics) {
    if (parent == null) {
      return new Module(mutability);
    }
    Preconditions.checkArgument(parent.mutability().isFrozen(), "parent frame must be frozen");
    Preconditions.checkArgument(parent.universe == null);

    Map<String, Object> filteredBindings = new LinkedHashMap<>();
    Map<String, FlagGuardedValue> restrictedBindings = new LinkedHashMap<>();

    for (Map.Entry<String, Object> binding : parent.bindings.entrySet()) {
      if (binding.getValue() instanceof FlagGuardedValue) {
        FlagGuardedValue val = (FlagGuardedValue) binding.getValue();
        if (val.isObjectAccessibleUsingSemantics(semantics)) {
          filteredBindings.put(binding.getKey(), val.getObject(semantics));
        } else {
          restrictedBindings.put(binding.getKey(), val);
        }
      } else {
        filteredBindings.put(binding.getKey(), binding.getValue());
      }
    }

    restrictedBindings.putAll(parent.restrictedBindings);

    return new Module(
        mutability, null /*parent */, parent.label, filteredBindings, restrictedBindings);
  }

  private void checkInitialized() {
    Preconditions.checkNotNull(mutability, "Attempted to use Frame before initializing it");
  }

  public void initialize(
      Mutability mutability,
      @Nullable Module universe,
      @Nullable Object label,
      Map<String, Object> bindings) {
    Preconditions.checkState(
        universe == null || universe.universe == null); // no more than 1 universe
    Preconditions.checkState(
        this.mutability == null, "Attempted to initialize an already initialized Frame");
    this.mutability = Preconditions.checkNotNull(mutability);
    this.universe = universe;
    if (label != null) {
      this.label = label;
    } else if (universe != null) {
      this.label = universe.label;
    } else {
      this.label = null;
    }
    this.bindings.putAll(bindings);
  }

  /**
   * Returns a new {@link Module} with the same fields, except that {@link #label} is set to the
   * given value. The label associated with each function (frame) on the stack is accessible using
   * {@link #getLabel}, and is included in the result of {@code str(fn)} where {@code fn} is a
   * StarlarkFunction.
   */
  public Module withLabel(Object label) {
    checkInitialized();
    return new Module(mutability, /*universe*/ null, label, bindings, /*restrictedBindings*/ null);
  }

  /** Returns the {@link Mutability} of this {@link Module}. */
  public Mutability mutability() {
    checkInitialized();
    return mutability;
  }

  /**
   * Returns the parent {@link Module}, if it exists.
   *
   * <p>TODO(laurentlb): Should be called getUniverse.
   */
  @Nullable
  public Module getParent() {
    checkInitialized();
    return universe;
  }

  /**
   * Returns the label (an optional piece of metadata) associated with this {@code Module}. (For
   * Bazel LOADING threads, this is the build label of its BUILD or .bzl file.)
   */
  @Nullable
  public Object getLabel() {
    checkInitialized();
    return label;
  }

  /**
   * Returns a map of direct bindings of this {@link Module}, ignoring universe.
   *
   * <p>The bindings are returned in a deterministic order (for a given sequence of initial values
   * and updates).
   *
   * <p>For efficiency an unmodifiable view is returned. Callers should assume that the view is
   * invalidated by any subsequent modification to the {@link Module}'s bindings.
   */
  public Map<String, Object> getBindings() {
    checkInitialized();
    return Collections.unmodifiableMap(bindings);
  }

  /**
   * Returns a map of bindings that are exported (i.e. symbols declared using `=` and `def`, but not
   * `load`).
   */
  // TODO(adonovan): whether bindings are exported should be decided by the resolver;
  // non-exported bindings should never be added to the module.
  public Map<String, Object> getExportedBindings() {
    checkInitialized();
    ImmutableMap.Builder<String, Object> result = new ImmutableMap.Builder<>();
    for (Map.Entry<String, Object> entry : bindings.entrySet()) {
      if (exportedBindings.contains(entry.getKey())) {
        result.put(entry);
      }
    }
    return result.build();
  }

  @Override
  public Set<String> getNames() {
    return getTransitiveBindings().keySet();
  }

  @Override
  public String getUndeclaredNameError(StarlarkSemantics semantics, String name) {
    FlagGuardedValue v = restrictedBindings.get(name);
    return v == null ? null : v.getErrorFromAttemptingAccess(semantics, name);
  }

  /** Returns an environment containing both module and predeclared bindings. */
  // TODO(adonovan): eliminate in favor of explicit module vs. predeclared operations.
  public Map<String, Object> getTransitiveBindings() {
    checkInitialized();
    // Can't use ImmutableMap.Builder because it doesn't allow duplicates.
    LinkedHashMap<String, Object> collectedBindings = new LinkedHashMap<>();
    if (universe != null) {
      collectedBindings.putAll(universe.getTransitiveBindings());
    }
    collectedBindings.putAll(getBindings());
    return collectedBindings;
  }

  /**
   * Returns the value of the specified module variable, or null if not bound. Does not look in the
   * predeclared environment.
   */
  public Object lookup(String varname) {
    checkInitialized();
    return bindings.get(varname);
  }

  /**
   * Returns the value of the named variable in the module environment, or if not bound there, in
   * the predeclared environment, or if not bound there, null.
   */
  public Object get(String varname) {
    // TODO(adonovan): delete this whole function, and getTransitiveBindings.
    // With proper resolution, the interpreter will know whether
    // to look in the module or the predeclared/universal environment.
    checkInitialized();
    Object val = bindings.get(varname);
    if (val != null) {
      return val;
    }
    if (universe != null) {
      return universe.get(varname);
    }
    return null;
  }

  /** Updates a binding in the module environment. */
  public void put(String varname, Object value) throws MutabilityException {
    Preconditions.checkNotNull(value, "Module.put(%s, null)", varname);
    checkInitialized();
    if (mutability.isFrozen()) {
      throw new MutabilityException("trying to mutate a frozen module");
    }
    bindings.put(varname, value);
  }

  @Override
  public String toString() {
    // TODO(adonovan): use the file name of the module (not visible to Starlark programs).
    if (mutability == null) {
      return "<Uninitialized Module>";
    } else {
      return String.format("<Module%s>", mutability());
    }
  }
}
