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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.SkylarkType.SkylarkFunctionType;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.Stack;

/**
 * An Environment for the semantic checking of Skylark files.
 *
 * @see Statement#validate
 * @see Expression#validate
 */
public class ValidationEnvironment {

  private final ValidationEnvironment parent;

  private Map<String, SkylarkType> variableTypes = new HashMap<>();

  private Map<String, Location> variableLocations = new HashMap<>();

  private Set<String> readOnlyVariables = new HashSet<>();

  // A stack of variable-sets which are read only but can be assigned in different
  // branches of if-else statements.
  private Stack<Set<String>> futureReadOnlyVariables = new Stack<>();

  // The function we are currently validating.
  private SkylarkFunctionType currentFunction;

  // Whether this validation environment is not modified therefore clonable or not.
  private boolean clonable;

  public ValidationEnvironment(Map<String, SkylarkType> builtinVariableTypes) {
    parent = null;
    variableTypes = new HashMap<>(builtinVariableTypes);
    readOnlyVariables.addAll(builtinVariableTypes.keySet());
    clonable = true;
  }

  private ValidationEnvironment(Map<String, SkylarkType> builtinVariableTypes,
      Set<String> readOnlyVariables) {
    parent = null;
    this.variableTypes = new HashMap<>(builtinVariableTypes);
    this.readOnlyVariables = new HashSet<>(readOnlyVariables);
    clonable = false;
  }

  // ValidationEnvironment for a new Environment()
  private static ImmutableMap<String, SkylarkType> globalTypes =
      new ImmutableMap.Builder<String, SkylarkType> ()
      .put("False", SkylarkType.BOOL).put("True", SkylarkType.BOOL)
      .put("None", SkylarkType.TOP).build();

  public ValidationEnvironment() {
    this(globalTypes);
  }

  @Override
  public ValidationEnvironment clone() {
    Preconditions.checkState(clonable);
    return new ValidationEnvironment(variableTypes, readOnlyVariables);
  }

  /**
   * Creates a local ValidationEnvironment to validate user defined function bodies.
   */
  public ValidationEnvironment(ValidationEnvironment parent, SkylarkFunctionType currentFunction) {
    // Don't copy readOnlyVariables: Variables may shadow global values.
    this.parent = parent;
    this.variableTypes = new HashMap<String, SkylarkType>();
    this.currentFunction = currentFunction;
    this.clonable = false;
  }

  /**
   * Returns true if this ValidationEnvironment is top level i.e. has no parent.
   */
  public boolean isTopLevel() {
    return parent == null;
  }

  /**
   * Declare a variable and add it to the environment.
   */
  public void declare(String varname, Location location)
      throws EvalException {
    checkReadonly(varname, location);
    if (parent == null) {  // top-level values are immutable
      readOnlyVariables.add(varname);
      if (!futureReadOnlyVariables.isEmpty()) {
        // Currently validating an if-else statement
        futureReadOnlyVariables.peek().add(varname);
      }
    }
    variableTypes.put(varname, SkylarkType.UNKNOWN);
    variableLocations.put(varname, location);
    clonable = false;
  }

  private void checkReadonly(String varname, Location location) throws EvalException {
    if (readOnlyVariables.contains(varname)) {
      throw new EvalException(location, String.format("Variable %s is read only", varname));
    }
  }

  /**
   * Returns true if the symbol exists in the validation environment.
   */
  public boolean hasSymbolInEnvironment(String varname) {
    return variableTypes.containsKey(varname)
        || topLevel().variableTypes.containsKey(varname);
  }

  /**
   * Returns the type of the existing variable.
   */
  public SkylarkType getVartype(String varname) {
    SkylarkType type = variableTypes.get(varname);
    if (type == null && parent != null) {
      type = parent.getVartype(varname);
    }
    return Preconditions.checkNotNull(type,
        String.format("Variable %s is not found in the validation environment", varname));
  }

  public SkylarkFunctionType getCurrentFunction() {
    return currentFunction;
  }

  private ValidationEnvironment topLevel() {
    return Preconditions.checkNotNull(parent == null ? this : parent);
  }

  /**
   * Adds a user defined function to the validation environment is not exists.
   */
  public void updateFunction(String name, SkylarkFunctionType type, Location loc)
      throws EvalException {
    checkReadonly(name, loc);
    if (variableTypes.containsKey(name)) {
      throw new EvalException(loc, "function " + name + " already exists");
    }
    variableTypes.put(name, type);
    clonable = false;
  }

  /**
   * Starts a session with temporarily disabled readonly checking for variables between branches.
   * This is useful to validate control flows like if-else when we know that certain parts of the
   * code cannot both be executed. 
   */
  public void startTemporarilyDisableReadonlyCheckSession() {
    futureReadOnlyVariables.add(new HashSet<String>());
    clonable = false;
  }

  /**
   * Finishes the session with temporarily disabled readonly checking.
   */
  public void finishTemporarilyDisableReadonlyCheckSession() {
    Set<String> variables = futureReadOnlyVariables.pop();
    readOnlyVariables.addAll(variables);
    if (!futureReadOnlyVariables.isEmpty()) {
      futureReadOnlyVariables.peek().addAll(variables);
    }
    clonable = false;
  }

  /**
   * Finishes a branch of temporarily disabled readonly checking.
   */
  public void finishTemporarilyDisableReadonlyCheckBranch() {
    readOnlyVariables.removeAll(futureReadOnlyVariables.peek());
    clonable = false;
  }
}
