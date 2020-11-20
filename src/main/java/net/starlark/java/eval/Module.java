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
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
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
 * records the module's build label (such as "//dir:file.bzl").
 *
 * <p>Use {@link #create} to create a {@link Module} with no predeclared bindings other than the
 * universal ones. Use {@link #withPredeclared(StarlarkSemantics, Map)} to create a module with the
 * predeclared environment specified by the map, using the semantics to determine whether any
 * FlagGuardedValues in the map are enabled or disabled.
 */
public final class Module implements Resolver.Module {

  // The module's predeclared environment. Excludes UNIVERSE bindings.
  private ImmutableMap<String, Object> predeclared;

  // The module's global bindings, in order of creation.
  private final LinkedHashMap<String, Object> globals = new LinkedHashMap<>();

  // Names of globals that are exported and can be loaded from other modules.
  // TODO(adonovan): eliminate this field when the resolver does its job properly.
  final HashSet<String> exportedGlobals = new HashSet<>();

  // An optional piece of metadata associated with the module/file.
  // May be set after construction (too obscure to burden the constructors).
  // Its toString appears to Starlark in str(function): "<function f from ...>".
  @Nullable private Object clientData;

  private Module(ImmutableMap<String, Object> predeclared) {
    this.predeclared = predeclared;
  }

  /**
   * Constructs a Module with the specified predeclared bindings, filtered by the semantics, in
   * addition to the standard environment, {@link Starlark#UNIVERSE}.
   */
  public static Module withPredeclared(
      StarlarkSemantics semantics, Map<String, Object> predeclared) {
    return new Module(filter(predeclared, semantics));
  }

  /**
   * Creates a module with no predeclared bindings other than the standard environment, {@link
   * Starlark#UNIVERSE}.
   */
  public static Module create() {
    return new Module(/*predeclared=*/ ImmutableMap.of());
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

  /**
   * Returns a map in which each semantics-enabled FlagGuardedValue has been replaced by the value
   * it guards. Disabled FlagGuardedValues are left in place, and should be treated as unavailable.
   * The iteration order is unchanged.
   */
  private static ImmutableMap<String, Object> filter(
      Map<String, Object> predeclared, StarlarkSemantics semantics) {
    ImmutableMap.Builder<String, Object> filtered = ImmutableMap.builder();
    for (Map.Entry<String, Object> bind : predeclared.entrySet()) {
      Object v = bind.getValue();
      if (v instanceof FlagGuardedValue) {
        FlagGuardedValue fv = (FlagGuardedValue) bind.getValue();
        if (fv.isObjectAccessibleUsingSemantics(semantics)) {
          v = fv.getObject();
        }
      }
      filtered.put(bind.getKey(), v);
    }
    return filtered.build();
  }

  /**
   * Sets the client data (an arbitrary application-specific value) associated with the module. It
   * may be retrieved using {@link #getClientData}. Its {@code toString} form appears in the result
   * of {@code str(fn)} where {@code fn} is a StarlarkFunction: "<function f from ...>".
   */
  public void setClientData(@Nullable Object clientData) {
    this.clientData = clientData;
  }

  /**
   * Returns the client data associated with this module by a prior call to {@link #setClientData}.
   */
  @Nullable
  public Object getClientData() {
    return clientData;
  }

  /** Returns the value of a predeclared (or universal) binding in this module. */
  Object getPredeclared(String name) {
    Object v = predeclared.get(name);
    if (v != null) {
      return v;
    }
    return Starlark.UNIVERSE.get(name);
  }

  /**
   * Returns this module's additional predeclared bindings. (Excludes {@link Starlark#UNIVERSE}.)
   *
   * <p>The map reflects any semantics-based filtering of FlagGuardedValues done by {@link
   * #withPredeclared}: enabled FlagGuardedValues are replaced by their underlying value.
   */
  public ImmutableMap<String, Object> getPredeclaredBindings() {
    return predeclared;
  }

  /**
   * Returns a read-only view of this module's global bindings.
   *
   * <p>The bindings are returned in a deterministic order (for a given sequence of initial values
   * and updates).
   */
  public Map<String, Object> getGlobals() {
    return Collections.unmodifiableMap(globals);
  }

  /**
   * Returns a map of bindings that are exported (i.e. symbols declared using `=` and `def`, but not
   * `load`).
   */
  // TODO(adonovan): whether bindings are exported should be decided by the resolver;
  //  non-exported bindings should never be added to the module.  Delete this,
  //  once loads bind locally (then all globals will be exported).
  public ImmutableMap<String, Object> getExportedGlobals() {
    ImmutableMap.Builder<String, Object> result = new ImmutableMap.Builder<>();
    for (Map.Entry<String, Object> entry : globals.entrySet()) {
      if (exportedGlobals.contains(entry.getKey())) {
        result.put(entry);
      }
    }
    return result.build();
  }

  /** Implements the resolver's module interface. */
  @Override
  public Set<String> getNames() {
    // TODO(adonovan): for now, the resolver treats all predeclared/universe
    //  and global names as one bucket (Scope.PREDECLARED). Fix that.
    // TODO(adonovan): opt: change the resolver to request names on
    //  demand to avoid all this set copying.
    HashSet<String> names = new HashSet<>();
    names.addAll(Starlark.UNIVERSE.keySet());
    for (Map.Entry<String, Object> bind : getPredeclaredBindings().entrySet()) {
      if (bind.getValue() instanceof FlagGuardedValue) {
        continue; // disabled
      }
      names.add(bind.getKey());
    }
    names.addAll(globals.keySet());
    return names;
  }

  @Override
  @Nullable
  public String getUndeclaredNameError(String name) {
    Object v = getPredeclared(name);
    return v instanceof FlagGuardedValue
        ? ((FlagGuardedValue) v).getErrorFromAttemptingAccess(name)
        : null;
  }

  /**
   * Returns the value of the specified global variable, or null if not bound. Does not look in the
   * predeclared environment.
   */
  public Object getGlobal(String name) {
    return globals.get(name);
  }

  /** Updates a global binding in the module environment. */
  public void setGlobal(String name, Object value) {
    Preconditions.checkNotNull(value, "Module.setGlobal(%s, null)", name);
    globals.put(name, value);
  }

  @Override
  public String toString() {
    return String.format("<module %s>", clientData == null ? "?" : clientData);
  }
}
