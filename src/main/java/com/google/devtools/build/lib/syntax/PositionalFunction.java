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

import com.google.devtools.build.lib.packages.Type.ConversionException;

import java.util.List;
import java.util.Map;

/**
 * Abstract implementation of Function for functions that only accept positional
 * parameters.
 */
public abstract class PositionalFunction extends AbstractFunction {

  private final int minArgs;
  private final int maxArgs;

  /**
   * Constructs a function called "name" that accepts between "minArgs" and
   * "maxArgs" positional arguments (inclusive), and no keyword arguments, and
   * handles error conditions appropriately.
   */
  public PositionalFunction(String name, int minArgs, int maxArgs) {
    super(name);
    this.minArgs = minArgs;
    this.maxArgs = maxArgs;
  }

  @Override
  public Object call(List<Object> args,
                     Map<String, Object> kwargs,
                     FuncallExpression ast,
                     Environment env)
      throws EvalException {
    if (kwargs.size() > 0) {
      throw new EvalException(ast.getLocation(),
          getName() + " does not accept keyword arguments");
    }
    int numArgs = args.size();
    if (numArgs < minArgs || numArgs > maxArgs) {
      String numArgsMessage = minArgs == maxArgs
          ? ("exactly " + minArgs)
          : ("between " + minArgs + " and " + maxArgs);
      String message = getName() + " requires " + numArgsMessage
          + " argument(s); " + numArgs + " supplied";
      throw new EvalException(ast.getLocation(), message);
    }
    try {
      return call(args, ast, env);
    } catch (ConversionException e) {
      throw new EvalException(ast.getLocation(), e.getMessage());
    }
  }

  /**
   * Like Function.call, but with an empty set of keyword parameters, and the
   * length of args is guaranteed to be in range [minArgs, maxArgs].
   */
  public abstract Object call(List<Object> args, FuncallExpression ast, Environment env)
      throws EvalException, ConversionException;

  /**
   * Same as {@link PositionalFunction} except its call method doesn't pass the Environment.
   */
  public abstract static class SimplePositionalFunction extends PositionalFunction {

    public SimplePositionalFunction(String name, int minArgs, int maxArgs) {
      super(name, minArgs, maxArgs);
    }

    @Override
    public Object call(List<Object> args, FuncallExpression ast, Environment env)
        throws ConversionException, EvalException {
      return call(args, ast);
    }

    /**
     * Same as {@link PositionalFunction#call} except with no Environment.
     */
    public abstract Object call(List<Object> args, FuncallExpression ast)
        throws ConversionException, EvalException;
  }
}
