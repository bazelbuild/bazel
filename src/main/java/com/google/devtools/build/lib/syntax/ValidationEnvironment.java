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

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * An Environment for the semantic checking of Skylark files.
 *
 * @see Statement#validate
 * @see Expression#validate
 */
public class ValidationEnvironment {

  private Map<String, SkylarkType> variableTypes = new HashMap<>();

  private Map<String, Location> variableLocations = new HashMap<>();

  private Set<String> readOnlyVariables = new HashSet<>();

  // The function we are currently validating.
  private String currentFunction;

  // Whether this validation environment is not modified therefore clonable or not.
  private boolean clonable;

  public ValidationEnvironment(ImmutableMap<String, SkylarkType> builtinVariableTypes) {
    variableTypes.putAll(builtinVariableTypes);
    readOnlyVariables.addAll(builtinVariableTypes.keySet());
    clonable = true;
  }

  private ValidationEnvironment(
      Map<String, SkylarkType> variableTypes, Set<String> readOnlyVariables) {
    this.variableTypes = new HashMap<>(variableTypes);
    this.readOnlyVariables = new HashSet<>(readOnlyVariables);
    clonable = false;
  }

  @Override
  public ValidationEnvironment clone() {
    Preconditions.checkState(clonable);
    return new ValidationEnvironment(variableTypes, readOnlyVariables);
  }

  /**
   * Updates the variable type if the new type is "stronger" then the old one.
   * The old and the new vartype has to be compatible, otherwise an EvalException is thrown.
   * The new type is stronger if the old one doesn't exist or unknown.
   */
  public void update(String varname, SkylarkType newVartype, Location location)
      throws EvalException {
    if (readOnlyVariables.contains(varname)) {
      throw new EvalException(location, String.format("Variable %s is read only", varname));
    }
    SkylarkType oldVartype = variableTypes.get(varname);
    if (oldVartype != null) {
      newVartype = oldVartype.infer(newVartype, "variable '" + varname + "'",
          location, variableLocations.get(varname));
    }
    variableTypes.put(varname, newVartype);
    variableLocations.put(varname, location);
    clonable = false;
  }

  public void checkIterable(SkylarkType type, Location loc) throws EvalException {
    if (type == SkylarkType.UNKNOWN) {
      // Until all the language is properly typed, we ignore Object types.
      return;
    }
    if (!Iterable.class.isAssignableFrom(type.getType())
        && !Map.class.isAssignableFrom(type.getType())) {
      throw new EvalException(loc,
          "type '" + EvalUtils.getDataTypeNameFromClass(type.getType()) + "' is not iterable");
    }
  }

  /**
   * Returns true if the variable exists.
   */
  public boolean hasVariable(String varname) {
    return variableTypes.containsKey(varname);
  }

  /**
   * Returns the type of the existing variable.
   */
  public SkylarkType getVartype(String varname) {
    return Preconditions.checkNotNull(variableTypes.get(varname),
        String.format("Variable %s is not found in the validation environment", varname));
  }

  public void setCurrentFunction(String fct) {
    currentFunction = fct;
    clonable = false;
  }

  public String getCurrentFunction() {
    return currentFunction;
  }
}
