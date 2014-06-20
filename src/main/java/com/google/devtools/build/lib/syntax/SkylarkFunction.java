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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Type.ConversionException;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ExecutionException;

/**
 * A function class for Skylark built in functions. Supports mandatory and optional arguments.
 * All usable arguments have to be specified. In case of ambiguous arguments (a parameter is
 * specified as positional and keyword arguments in the function call) an exception is thrown.
 */
public abstract class SkylarkFunction extends AbstractFunction {

  private ImmutableList<String> parameters;
  private int mandatoryParamNum;
  private boolean configured = false;

  /**
   * Creates a SkylarkFunction with the given name. 
   */
  public SkylarkFunction(String name) {
    super(name);
  }

  /**
   * Configures the parameter of this Skylark function using the annotation.
   */
  @VisibleForTesting
  public void configure(SkylarkBuiltin annotation) {
    Preconditions.checkState(!configured);
    Preconditions.checkArgument(getName().equals(annotation.name()));
    mandatoryParamNum = 0;
    ImmutableList.Builder<String> paramListBuilder = ImmutableList.builder();
    for (SkylarkBuiltin.Param param : annotation.mandatoryParams()) {
      paramListBuilder.add(param.name());
      mandatoryParamNum++;
    }
    for (SkylarkBuiltin.Param param : annotation.optionalParams()) {
      paramListBuilder.add(param.name());
    }
    parameters = paramListBuilder.build();
    configured = true;
  }

  /**
   * Returns true if the SkylarkFunction is configured.
   */
  public boolean isConfigured() {
    return configured;
  }

  @Override
  public Object call(List<Object> args,
                     Map<String, Object> kwargs,
                     FuncallExpression ast,
                     Environment env)
      throws EvalException, InterruptedException {

    Preconditions.checkState(configured);
    try {
      ImmutableMap.Builder<String, Object> arguments = new ImmutableMap.Builder<>();
      int maxParamNum = parameters.size();
      int paramNum = args.size() + kwargs.size();
  
      if (paramNum < mandatoryParamNum || paramNum > maxParamNum) {
        throw new EvalException(ast.getLocation(),
            String.format("incorrect number of arguments %s (expected %s - %s)",
                paramNum, mandatoryParamNum, maxParamNum));
      }
  
      for (int i = 0; i < mandatoryParamNum; i++) {
        Preconditions.checkState(i < args.size() || kwargs.containsKey(parameters.get(i)),
            String.format("missing mandatory parameter: %s", parameters.get(i)));
      }
  
      for (int i = 0; i < args.size(); i++) {
        arguments.put(parameters.get(i), args.get(i));
      }
  
      for (Entry<String, Object> kwarg : kwargs.entrySet()) {
        int idx = parameters.indexOf(kwarg.getKey()); 
        if (idx < 0) {
          throw new EvalException(ast.getLocation(),
              String.format("unknown keyword argument: %s", kwarg.getKey()));
        }
        if (idx < args.size()) {
          throw new EvalException(ast.getLocation(),
              String.format("ambiguous argument: %s", kwarg.getKey()));
        }
        arguments.put(kwarg.getKey(), kwarg.getValue());
      }

      return call(arguments.build(), ast, env);
    } catch (ConversionException | IllegalArgumentException | IllegalStateException
        | ClassCastException | ClassNotFoundException | ExecutionException e) {
      throw new EvalException(ast.getLocation(), e.getMessage());
    }
  }

  /**
   * The actual function call. All positional and keyword arguments are put in the
   * arguments map.
   */
  protected abstract Object call(
      Map<String, Object> arguments, FuncallExpression ast, Environment env) throws EvalException,
      ConversionException,
      IllegalArgumentException,
      IllegalStateException,
      ClassCastException,
      ClassNotFoundException,
      ExecutionException;

  /**
   * An intermediate class to provide a simpler interface for Skylark functions.
   */
  public abstract static class SimpleSkylarkFunction extends SkylarkFunction {

    public SimpleSkylarkFunction(String name) {
      super(name);
    }

    @Override
    protected final Object call(
        Map<String, Object> arguments, FuncallExpression ast, Environment env) throws EvalException,
        ConversionException,
        IllegalArgumentException,
        IllegalStateException,
        ClassCastException,
        ExecutionException {
      return call(arguments, ast.getLocation());
    }

    /**
     * The actual function call. All positional and keyword arguments are put in the
     * arguments map.
     */
    protected abstract Object call(Map<String, Object> arguments, Location loc)
        throws EvalException,
        ConversionException,
        IllegalArgumentException,
        IllegalStateException,
        ClassCastException,
        ExecutionException;
  }

  /**
   * Collects the SkylarkFunctions from the fields of the class of the object parameter
   * and adds them into the builder.
   */
  public static void collectSkylarkFunctionsFromFields(
      Object object, ImmutableList.Builder<Function> builder) {
    for (Field field : object.getClass().getDeclaredFields()) {
      if (SkylarkFunction.class.isAssignableFrom(field.getType())
          && field.isAnnotationPresent(SkylarkBuiltin.class)) {
        try {
          field.setAccessible(true);
          SkylarkFunction function = (SkylarkFunction) field.get(object);
          SkylarkBuiltin annotation = field.getAnnotation(SkylarkBuiltin.class);
          // TODO(bazel-team): we need this because of the static functions. We need
          // static functions because of the testing. The tests use a mixture of Skylark
          // and non Skylark rules. this causes the problem. As soon as we have only
          // Skylark rules in the SkylarkTests we can clean this up.
          if (!function.isConfigured()) {
            function.configure(annotation);
          }
          builder.add(function);
        } catch (IllegalArgumentException | IllegalAccessException e) {
          // This should never happen.
          throw new RuntimeException(e);
        }
      }
    }
  }

  /**
   * Collects the SkylarkFunctions from the fields of the class of the object parameter
   * and adds their class and their corresponding return value to the builder.
   */
  public static void collectSkylarkFunctionReturnTypesFromFields(
      Class<?> classObject, ImmutableMap.Builder<String, Class<?>> builder) {
    for (Field field : classObject.getDeclaredFields()) {
      if (SkylarkFunction.class.isAssignableFrom(field.getType())
          && field.isAnnotationPresent(SkylarkBuiltin.class)) {
        try {
          field.setAccessible(true);
          SkylarkBuiltin annotation = field.getAnnotation(SkylarkBuiltin.class);
          // TODO(bazel-team): infer the correct types.
          builder.put(annotation.name(), Object.class);
          builder.put(annotation.name() + ".return", Object.class);
        } catch (IllegalArgumentException e) {
          // This should never happen.
          throw new RuntimeException(e);
        }
      }
    }
  }
}
