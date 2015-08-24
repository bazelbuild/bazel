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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * The BUILD environment.
 */
public class Environment {

  @SkylarkSignature(name = "True", returnType = Boolean.class,
      doc = "Literal for the boolean true.")
  private static final Boolean TRUE = true;

  @SkylarkSignature(name = "False", returnType = Boolean.class,
      doc = "Literal for the boolean false.")
  private static final Boolean FALSE = false;

  @SkylarkSignature(name = "PACKAGE_NAME", returnType = String.class,
      doc = "The name of the package the rule or build extension is called from. "
          + "For example, in the BUILD file <code>some/package/BUILD</code>, its value "
          + "will be <code>some/package</code>. "
          + "This variable is special, because its value comes from outside of the extension "
          + "module (it comes from the BUILD file), so it can only be accessed in functions "
          + "(transitively) called from BUILD files. For example:<br>"
          + "<pre class=language-python>def extension():\n"
          + "  return PACKAGE_NAME</pre>"
          + "In this case calling <code>extension()</code> works from the BUILD file (if the "
          + "function is loaded), but not as a top level function call in the extension module.")
  public static final String PKG_NAME = "PACKAGE_NAME";

  /**
   * There should be only one instance of this type to allow "== None" tests.
   */
  @Immutable
  public static final class NoneType {
    @Override
    public String toString() { return "None"; }
    private NoneType() {}
  }

  @SkylarkSignature(name = "None", returnType = NoneType.class, doc = "Literal for the None value.")
  public static final NoneType NONE = new NoneType();

  protected final Map<String, Object> env = new HashMap<>();

  // BaseFunctions with namespaces. Works only in the global environment.
  protected final Map<Class<?>, Map<String, BaseFunction>> functions = new HashMap<>();

  /**
   * The parent environment. For Skylark it's the global environment,
   * used for global read only variable lookup.
   */
  protected final Environment parent;

  /**
   * Map from a Skylark extension to an environment, which contains all symbols defined in the
   * extension.
   */
  protected Map<PathFragment, SkylarkEnvironment> importedExtensions;

  /**
   * A set of disabled namespaces propagating through function calling. This is needed because
   * UserDefinedFunctions lock the definition Environment which should be immutable.
   */
  protected Set<Class<?>> disabledNameSpaces = new HashSet<>();

  /**
   * A set of variables propagating through function calling. It's only used to call
   * native rules from Skylark build extensions.
   */
  protected Set<String> propagatingVariables = new HashSet<>();

  /**
   * An EventHandler for errors and warnings. This is not used in the BUILD language,
   * however it might be used in Skylark code called from the BUILD language.
   */
  @Nullable protected EventHandler eventHandler;

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

  /**
   * Constructs an empty child environment with an EventHandler.
   */
  public Environment(Environment parent, EventHandler eventHandler) {
    this(parent);
    this.eventHandler = Preconditions.checkNotNull(eventHandler);
  }

  public EventHandler getEventHandler() {
    return eventHandler;
  }

  // Sets up the global environment
  private void setupGlobal() {
    // In Python 2.x, True and False are global values and can be redefined by the user.
    // In Python 3.x, they are keywords. We implement them as values, for the sake of
    // simplicity. We define them as Boolean objects.
    update("False", FALSE);
    update("True", TRUE);
    update("None", NONE);
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
   * Updates the value of variable "varname" in the environment, corresponding
   * to an {@link AssignmentStatement}.
   */
  public void update(String varname, Object value) {
    Preconditions.checkNotNull(value, "update(value == null)");
    env.put(varname, value);
  }

  /**
   * Same as {@link #update}, but also marks the variable propagating, meaning it will
   * be present in the execution environment of a UserDefinedFunction called from this
   * Environment. Using this method is discouraged.
   */
  public void updateAndPropagate(String varname, Object value) {
    update(varname, value);
    propagatingVariables.add(varname);
  }

  /**
   * Remove the variable from the environment, returning
   * any previous mapping (null if there was none).
   */
  public Object remove(String varname) {
    return env.remove(varname);
  }

  /**
   * Returns the (immutable) set of names of all variables directly defined in this environment.
   */
  public Set<String> getDirectVariableNames() {
    return env.keySet();
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
    for (String key : keys) {
      out.append(key).append(" -> ").append(env.get(key)).append(", ");
    }
    out.append("}");
    if (parent != null) {
      out.append("=>");
      out.append(parent);
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

  /**
   * An exception thrown when an attempt is made to import a symbol from a file
   * that was not properly loaded.
   */
  public static class LoadFailedException extends Exception {
    LoadFailedException(String file) {
      super("file '" + file + "' was not correctly loaded. Make sure the 'load' statement appears "
          + "in the global scope, in the BUILD file");
    }
  }

  public void setImportedExtensions(Map<PathFragment, SkylarkEnvironment> importedExtensions) {
    this.importedExtensions = importedExtensions;
  }

  public void importSymbol(PathFragment extension, Identifier symbol, String nameInLoadedFile)
      throws NoSuchVariableException, LoadFailedException {
    if (!importedExtensions.containsKey(extension)) {
      throw new LoadFailedException(extension.toString());
    }

    Object value = importedExtensions.get(extension).lookup(nameInLoadedFile);
    if (!isSkylarkEnabled()) {
      value = SkylarkType.convertFromSkylark(value);
    }

    update(symbol.getName(), value);
  }

  /**
   * Registers a function with namespace to this global environment.
   */
  public void registerFunction(Class<?> nameSpace, String name, BaseFunction function) {
    nameSpace = getCanonicalRepresentation(nameSpace);
    Preconditions.checkArgument(parent == null);
    if (!functions.containsKey(nameSpace)) {
      functions.put(nameSpace, new HashMap<String, BaseFunction>());
    }
    functions.get(nameSpace).put(name, function);
  }

  private Map<String, BaseFunction> getNamespaceFunctions(Class<?> nameSpace) {
    nameSpace = getCanonicalRepresentation(nameSpace);
    if (disabledNameSpaces.contains(nameSpace)
        || (parent != null && parent.disabledNameSpaces.contains(nameSpace))) {
      return null;
    }
    Environment topLevel = this;
    while (topLevel.parent != null) {
      topLevel = topLevel.parent;
    }
    return topLevel.functions.get(nameSpace);
  }

  /**
   * Returns the canonical representation of the given class, i.e. the super class for which any
   * functions were registered.
   *
   * <p>Currently, this is only necessary for mapping the different subclasses of {@link
   * java.util.Map} to the interface.
   */
  private Class<?> getCanonicalRepresentation(Class<?> clazz) {
    return Map.class.isAssignableFrom(clazz) ? Map.class : clazz;
  }

  /**
   * Returns the function of the namespace of the given name or null of it does not exists.
   */
  public BaseFunction getFunction(Class<?> nameSpace, String name) {
    Map<String, BaseFunction> nameSpaceFunctions = getNamespaceFunctions(nameSpace);
    return nameSpaceFunctions != null ? nameSpaceFunctions.get(name) : null;
  }

  /**
   * Returns the function names registered with the namespace.
   */
  public Set<String> getFunctionNames(Class<?> nameSpace) {
    Map<String, BaseFunction> nameSpaceFunctions = getNamespaceFunctions(nameSpace);
    return nameSpaceFunctions != null ? nameSpaceFunctions.keySet() : ImmutableSet.<String>of();
  }

  /**
   * Return the current stack trace (list of functions).
   */
  public ImmutableList<BaseFunction> getStackTrace() {
    // Empty list, since this environment does not allow function definition
    // (see SkylarkEnvironment)
    return ImmutableList.of();
  }
}
