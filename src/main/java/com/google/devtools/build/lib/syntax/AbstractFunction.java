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

import java.util.List;
import java.util.Map;

/**
 * Partial implementation of Function interface.
 */
public abstract class AbstractFunction implements Function {

  private final String name;

  protected AbstractFunction(String name) {
    this.name = name;
  }

  /**
   * Returns the name of this function.
   */
  @Override
  public String getName() {
    return name;
  }

  @Override
  public Class<?> getObjectType() {
    return null;
  }

  @Override public String toString() {
    return name;
  }

  /**
   * Abstract implementation of Function that accepts no parameters.
   */
  public abstract static class NoArgFunction extends AbstractFunction {

    public NoArgFunction(String name) {
      super(name);
    }

    @Override
    public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
        Environment env) throws EvalException, InterruptedException {
      if (args.size() != 1 || kwargs.size() != 0) {
        throw new EvalException(ast.getLocation(), "Invalid number of arguments (expected 0)");
      }
      return call(args.get(0), ast, env);
    }

    public abstract Object call(Object self, FuncallExpression ast, Environment env)
        throws EvalException, InterruptedException;
  }
}
