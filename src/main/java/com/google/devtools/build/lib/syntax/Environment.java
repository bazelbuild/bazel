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
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * The BUILD environment.
 */
public class Environment {

  protected final Map<String, Object> env = new HashMap<>();

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
   * A set of variables propagating through function calling. It's only used to call
   * native rules from Skylark build extensions.
   */
  protected Set<String> propagatingVariables = new HashSet<>();

  // Only used in the global environment.
  // TODO(bazel-team): make this a final field part of constructor.
  private boolean isLoadingPhase = false;

  /**
   * Is this Environment being evaluated during the loading phase?
   * This is fixed during environment setup, and enables various functions
   * that are not available during the analysis phase.
   * @return true if this environment corresponds to code during the loading phase.
   */
  boolean isLoadingPhase() {
    return isLoadingPhase;
  }

  /**
   * Enable loading phase only functions in this Environment.
   * This should only be done during setup before code is evaluated.
   */
  public void setLoadingPhase() {
    isLoadingPhase = true;
  }

  /**
   * Checks that the current Evaluation context is in loading phase.
   * @param symbol name of the function being only authorized thus.
   */
  public void checkLoadingPhase(String symbol, Location loc) throws EvalException {
    if (!isLoadingPhase()) {
      throw new EvalException(loc, symbol + "() can only be called during the loading phase");
    }
  }

  /**
   * Is this a global environment?
   * @return true if this is a global (top-level) environment
   * as opposed to inside the body of a function
   */
  public boolean isGlobal() {
    return true;
  }

  /**
   * An EventHandler for errors and warnings. This is not used in the BUILD language,
   * however it might be used in Skylark code called from the BUILD language.
   */
  @Nullable protected EventHandler eventHandler;

  /**
   * A stack trace containing the current history of functions and the calling rule.
   *
   * <p>For the rule, the stack trace has two elements: one for the call to the rule in the BUILD
   * file and one for the actual rule implementation.
   */
  private Deque<StackTraceElement> stackTrace;

  /**
   * Constructs an empty root non-Skylark environment.
   * The root environment is also the global environment.
   */
  public Environment(Deque<StackTraceElement> stackTrace) {
    this.parent = null;
    this.importedExtensions = new HashMap<>();
    this.stackTrace = stackTrace;
    setupGlobal();
  }

  public Environment() {
    this(new LinkedList<StackTraceElement>());
  }

  /**
   * Constructs an empty child environment.
   */
  public Environment(Environment parent, Deque<StackTraceElement> stackTrace) {
    Preconditions.checkNotNull(parent);
    this.parent = parent;
    this.importedExtensions = new HashMap<>();
    this.stackTrace = stackTrace;
  }

  public Environment(Environment parent) {
    this(parent, new LinkedList<StackTraceElement>());
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
    update("False", Runtime.FALSE);
    update("True", Runtime.TRUE);
    update("None", Runtime.NONE);
  }

  public boolean isSkylark() {
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
  public Environment update(String varname, Object value) {
    Preconditions.checkNotNull(value, "update(value == null)");
    env.put(varname, value);
    return this;
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
    if (!isSkylark()) {
      value = SkylarkType.convertFromSkylark(value);
    }

    update(symbol.getName(), value);
  }

  public ImmutableList<StackTraceElement> getStackTrace() {
    return ImmutableList.copyOf(stackTrace);
  }

  protected Deque<StackTraceElement> getCopyOfStackTrace() {
    return new LinkedList<>(stackTrace);
  }

  /**
   * Adds the given element to the stack trace (iff the stack is empty) and returns whether it was
   * successful.
   */
  public boolean tryAddingStackTraceRoot(StackTraceElement element) {
    if (stackTrace.isEmpty()) {
      stackTrace.add(element);
      return true;
    }
    return false;
  }
  
  public void addToStackTrace(StackTraceElement element)    {
    stackTrace.add(element);
  }

  /**
   * Removes the only remaining element from the stack trace.
   *
   * <p>This particular element describes the outer-most calling function (usually a rule).
   *
   * <p> This method is required since {@link FuncallExpression} does not create a new {@link
   * Environment}, hence it has to add and remove its {@link StackTraceElement} from an existing
   * one.
   */
  public void removeStackTraceRoot() {
    Preconditions.checkArgument(stackTrace.size() == 1);
    stackTrace.clear();
  }

  public void removeStackTraceElement() {
    // TODO(fwe): find out why the precond doesn't work
    //    Preconditions.checkArgument(stackTrace.size() > 1);
    stackTrace.removeLast();
  }

  /**
   * Returns whether the given {@link BaseFunction} is part of this {@link Environment}'s stack
   * trace.
   */
  public boolean stackTraceContains(BaseFunction function) {
    for (StackTraceElement element : stackTrace) {
      if (element.hasFunction(function)) {
        return true;
      }
    }
    return false;
  }
}
