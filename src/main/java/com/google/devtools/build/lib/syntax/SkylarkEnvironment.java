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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;

import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

/**
 * The environment for Skylark.
 */
public class SkylarkEnvironment extends Environment {

  /**
   * This set contains the variable names of all the successful lookups from the global
   * environment. This is necessary because if in a function definition something
   * reads a global variable after which a local variable with the same name is assigned an
   * Exception needs to be thrown.
   */
  private final Set<String> readGlobalVariables = new HashSet<>();

  private ImmutableList<String> stackTrace;

  /**
   * Creates a Skylark Environment for function calling, from the global Environment of the
   * caller Environment (which must be a Skylark Environment).
   */
  public static SkylarkEnvironment createEnvironmentForFunctionCalling(
      SkylarkEnvironment callerEnv, UserDefinedFunction function) throws EvalException {
    if (callerEnv.getStackTrace().contains(function.getName())) {
      throw new EvalException(function.getLocation(), "Recursion was detected when calling '"
          + function.getName() + "' from '" + Iterables.getLast(callerEnv.getStackTrace()) + "'");
    }
    ImmutableList<String> stackTrace = new ImmutableList.Builder<String>()
        .addAll(callerEnv.getStackTrace())
        .add(function.getName())
        .build();
    return new SkylarkEnvironment(callerEnv, stackTrace);
  }

  private SkylarkEnvironment(SkylarkEnvironment callerEnv, ImmutableList<String> stackTrace) {
    super(callerEnv.getGlobalEnvironment());
    this.stackTrace = stackTrace;
  }

  /**
   * Creates a global SkylarkEnvironment.
   */
  public SkylarkEnvironment() {
    super();
    stackTrace = ImmutableList.of();
  }

  public SkylarkEnvironment(SkylarkEnvironment globalEnv) {
    super(globalEnv);
    stackTrace = ImmutableList.of();
  }

  public ImmutableList<String> getStackTrace() {
    return stackTrace;
  }

  /**
   * Clones this Skylark global environment.
   */
  public SkylarkEnvironment cloneEnv() {
    Preconditions.checkArgument(isGlobalEnvironment());
    SkylarkEnvironment newEnv = new SkylarkEnvironment();
    for (Entry<String, Object> entry : env.entrySet()) {
      newEnv.env.put(entry.getKey(), entry.getValue());
    }
    for (Map.Entry<Class<?>, Map<String, Function>> functionMap : functions.entrySet()) {
      newEnv.functions.put(functionMap.getKey(), functionMap.getValue());
    }
    return newEnv;
  }

  /**
   * Returns the global environment. Only works for Skylark environments. For the global Skylark
   * environment this method returns this Environment.
   */
  public SkylarkEnvironment getGlobalEnvironment() {
    // If there's a parent that's the global environment, otherwise this is.
    return parent != null ? (SkylarkEnvironment) parent : this;
  }

  /**
   * Returns true if this is a Skylark global environment.
   */
  public boolean isGlobalEnvironment() {
    return parent == null;
  }

  /**
   * Returns true if varname has been read as a global variable.
   */
  public boolean hasBeenReadGlobalVariable(String varname) {
    return readGlobalVariables.contains(varname);
  }

  @Override
  public boolean isSkylarkEnabled() {
    return true;
  }

  /**
   * @return the value from the environment whose name is "varname".
   * @throws NoSuchVariableException if the variable is not defined in the environment.
   */
  @Override
  public Object lookup(String varname) throws NoSuchVariableException {
    Object value = env.get(varname);
    if (value == null) {
      if (parent != null && parent.hasVariable(varname)) {
        readGlobalVariables.add(varname);
        return parent.lookup(varname);
      }
      throw new NoSuchVariableException(varname);
    }
    return value;
  }

  /**
   * Like <code>lookup(String)</code>, but instead of throwing an exception in
   * the case where "varname" is not defined, "defaultValue" is returned instead.
   */
  @Override
  public Object lookup(String varname, Object defaultValue) {
    Object value = env.get(varname);
    if (value == null) {
      if (parent != null && parent.hasVariable(varname)) {
        readGlobalVariables.add(varname);
        return parent.lookup(varname, defaultValue);
      }
      return defaultValue;
    }
    return value;
  }

  /**
   * Updates the value of variable "varname" in the environment, corresponding
   * to an AssignmentStatement.
   */
  @Override
  public void update(String varname, Object value) {
    Preconditions.checkNotNull(value, "update(value == null)");
    Preconditions.checkArgument(!isReadOnly(varname));
    env.put(varname, value);
  }

  public boolean isReadOnly(String varname) {
    return isGlobalEnvironment() && env.containsKey(varname);
  }

  /**
   * Returns the class of the variable or null if the variable does not exist. This function
   * works only in the local Environment, it doesn't check the global Environment.
   */
  public Class<?> getVariableType(String varname) {
    Object variable = env.get(varname);
    return variable != null ? EvalUtils.getSkylarkType(variable.getClass()) : null;
  }
}
