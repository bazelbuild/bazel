// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The BUILD environment.
 */
public class Environment {

  @SkylarkBuiltin(name = "True", doc = "Literal for the boolean true.")
  private static final boolean TRUE = true;

  @SkylarkBuiltin(name = "False", doc = "Literal for the boolean false.")
  private static final boolean FALSE = false;

  /**
   * There should be only one instance of this type to allow "== None" tests.
   */
  public static final class NoneType {
    @Override
    public String toString() { return "None"; }
    private NoneType() {}
  }

  @SkylarkBuiltin(name = "None", doc = "Literal for the None value.")
  public static final NoneType NONE = new NoneType();

  protected final Map<String, Object> env = new HashMap<>();

  // Functions with namespaces. Works only in the global environment.
  protected final Map<Class<?>, Map<String, Function>> functions = new HashMap<>();

  /**
   * The parent environment. For Skylark it's the global environment,
   * used for global read only variable lookup.
   */
  protected final Environment parent;

  /**
   * Map from a Skylark extension to an environment, which contains all symbols defined in the
   * extension.
   */
  protected Map<PathFragment, Environment> importedExtensions;

  /**
   * Constructs an empty root non-Skylark environment.
   * The root environment is also the global environment.
   */
  public Environment() {
    this.parent = null;
    this.importedExtensions = new HashMap<>();
    setupGlobal();
  }

  /**
   * Constructs an empty child environment.
   */
  public Environment(Environment parent) {
    Preconditions.checkNotNull(parent);
    this.parent = parent;
    this.importedExtensions = new HashMap<>();
  }

  // Sets up the global environment
  private void setupGlobal() {
    // In Python 2.x, True and False are global values and can be redefined by the user.
    // In Python 3.x, they are keywords. We implement them as values, for the sake of
    // simplicity. We define them as Boolean objects.
    env.put("False", FALSE);
    env.put("True", TRUE);
    env.put("None", NONE);
  }

  public boolean isSkylarkEnabled() {
    return false;
  }

  protected boolean hasVariable(String varname) {
    return env.containsKey(varname);
  }

  /**
   * @return the value from the environment whose name is "varname".
   * @throws NoSuchVariableException if the variable is not defined in the Environment.
   *
   */
  public Object lookup(String varname) throws NoSuchVariableException {
    Object value = env.get(varname);
    if (value == null) {
      if (parent != null) {
        return parent.lookup(varname);
      }
      throw new NoSuchVariableException(varname);
    }
    return value;
  }

  /**
   * Like <code>lookup(String)</code>, but instead of throwing an exception in
   * the case where "varname" is not defined, "defaultValue" is returned instead.
   *
   */
  public Object lookup(String varname, Object defaultValue) {
    Object value = env.get(varname);
    if (value == null) {
      if (parent != null) {
        return parent.lookup(varname, defaultValue);
      }
      return defaultValue;
    }
    return value;
  }

  /**
   * Returns all variables in the environment of the given type.
   */
  public <T> Map<String, T> getAll(Class<T> type) {
    ImmutableMap.Builder<String, T> builder = ImmutableMap.builder();
    for (Map.Entry<String, Object> var : env.entrySet()) {
      if (type.isAssignableFrom(var.getValue().getClass())) {
        builder.put(var.getKey(), type.cast(var.getValue()));
      }
    }
    return builder.build();
  }

  /**
   * Updates the value of variable "varname" in the environment, corresponding
   * to an {@link AssignmentStatement}.
   */
  public void update(String varname, Object value) {
    Preconditions.checkNotNull(value, "update(value == null)");
    env.put(varname, value);
  }

  /**
   * Remove the variable from the environment, returning
   * any previous mapping (null if there was none).
   */
  public Object remove(String varname) {
    return env.remove(varname);
  }

  /**
   * Returns the (immutable) set of names of all variables defined in this
   * environment. Exposed for testing; not very efficient!
   */
  @VisibleForTesting
  public Set<String> getVariableNames() {
    if (parent == null) {
      return env.keySet();
    } else {
      Set<String> vars = new HashSet<>();
      vars.addAll(env.keySet());
      vars.addAll(parent.getVariableNames());
      return vars;
    }
  }

  @Override
  public int hashCode() {
    throw new UnsupportedOperationException(); // avoid nondeterminism
  }

  @Override
  public boolean equals(Object that) {
    throw new UnsupportedOperationException();
  }

  @Override
  public String toString() {
    StringBuilder out = new StringBuilder();
    out.append("Environment{");
    List<String> keys = new ArrayList<>(env.keySet());
    Collections.sort(keys);
    for (String key: keys) {
      out.append(key).append(" -> ").append(env.get(key)).append(", ");
    }
    out.append("}");
    if (parent != null) {
      out.append("=>");
      out.append(parent.toString());
    }
    return out.toString();
  }

  /**
   * An exception thrown when an attempt is made to lookup a non-existent
   * variable in the environment.
   */
  public static class NoSuchVariableException extends Exception {
    NoSuchVariableException(String variable) {
      super("no such variable: " + variable);
    }
  }

  public void setImportedExtensions(Map<PathFragment, Environment> importedExtensions) {
    this.importedExtensions = importedExtensions;
  }

  public void importSymbol(PathFragment extension, String symbol) throws NoSuchVariableException {
    if (!importedExtensions.containsKey(extension)) {
      throw new NoSuchVariableException(extension.toString());
    }
    update(symbol, importedExtensions.get(extension).lookup(symbol));
  }

  /**
   * Registers a function with namespace to this global environment.
   */
  public void registerFunction(Class<?> nameSpace, String name, Function function) {
    Preconditions.checkArgument(parent == null);
    if (!functions.containsKey(nameSpace)) {
      functions.put(nameSpace, new HashMap<String, Function>());
    }
    functions.get(nameSpace).put(name, function);
  }

  /**
   * Returns the function of the namespace of the given name or null of it does not exists.
   */
  public Function getFunction(Class<?> nameSpace, String name) {
    Environment topLevel = this;
    while (topLevel.parent != null) {
      topLevel = topLevel.parent;
    }
    Map<String, Function> nameSpaceFunctions = topLevel.functions.get(nameSpace);
    return nameSpaceFunctions != null ? nameSpaceFunctions.get(name) : null;
  }
}
