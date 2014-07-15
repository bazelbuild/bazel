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

  private Map<String, Class<?>> variableTypes = new HashMap<>();

  private Map<String, Location> variableLocations = new HashMap<>();

  private Set<String> readOnlyVariables = new HashSet<>();

  // The function we are currently validating.
  private String currentFunction;

  // Whether this validation environment is not modified therefore clonable or not.
  private boolean clonable;

  public ValidationEnvironment(ImmutableMap<String, Class<?>> builtinVariableTypes) {
    variableTypes.putAll(builtinVariableTypes);
    readOnlyVariables.addAll(builtinVariableTypes.keySet());
    clonable = true;
  }

  private ValidationEnvironment(
      Map<String, Class<?>> variableTypes, Set<String> readOnlyVariables) {
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
  public void update(String varname, Class<?> newVartype, Location location) throws EvalException {
    if (readOnlyVariables.contains(varname)) {
      throw new EvalException(location, String.format("Variable %s is read only", varname));
    }
    if (!isVariableTypeCompatible(varname, newVartype)) {
      throw new EvalException(location, String.format("Incompatible variable types, "
          + "trying to assign type of %s to variable %s which is also a %s at %s",
          EvalUtils.getDataTypeNameFromClass(newVartype),
          varname,
          EvalUtils.getDataTypeNameFromClass(getVartype(varname)),
          variableLocations.get(varname)));
    }
    Class<?> oldVartype = variableTypes.get(varname);
    if (oldVartype == null || oldVartype.equals(Object.class)) {
      variableTypes.put(varname, newVartype);
      variableLocations.put(varname, location);
    }
    clonable = false;
  }

  public void checkIterable(Class<?> type, Location loc) throws EvalException {
    if (type.equals(Object.class)) {
      // Until all the language is properly typed, we ignore Object types.
      return;
    }
    if (!Iterable.class.isAssignableFrom(type) && !Map.class.isAssignableFrom(type)) {
      throw new EvalException(loc,
          "type '" + EvalUtils.getDataTypeNameFromClass(type) + "' is not iterable");
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
  public Class<?> getVartype(String varname) {
    return Preconditions.checkNotNull(variableTypes.get(varname),
        String.format("Variable %s is not found in the validation environment", varname));
  }

  /**
   * Returns true if the new vartype is compatible with the old one.
   */
  private boolean isVariableTypeCompatible(String varname, Class<?> newVartype) {
    if (newVartype.equals(Object.class)) {
      // Object.class means unknown variable type
      return true;
    }
    Class<?> oldVartype = variableTypes.get(varname);
    if (oldVartype == null) {
      return true;
    } else if (oldVartype.equals(Object.class)) {
        return true;
    } else {
        return newVartype.equals(oldVartype);
    }
  }

  public void setCurrentFunction(String fct) {
    currentFunction = fct;
    clonable = false;
  }

  public String getCurrentFunction() {
    return currentFunction;
  }
}
