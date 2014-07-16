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

package com.google.devtools.build.lib.rules;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.syntax.AbstractFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Function;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin;

import java.util.List;
import java.util.Map;

/**
 * A helper class to provide Attr module in Skylark.
 */
@SkylarkBuiltin(name = "Attr", doc = "A class to create rule attributes.")
public final class SkylarkAttr {
  // TODO(bazel-team): Expose all attr functions.
  // TODO(bazel-team): Better check the arguments.

  private static Object makeAttr(List<Object> args, Map<String, Object> kwargs, String type,
      FuncallExpression ast, Environment env) throws EvalException {
    if (args.size() != 1) {
      throw new EvalException(ast.getLocation(), "This function allows only keywords arguments");
    }
    try {
      return SkylarkRuleClassFunctions.createAttribute(type, kwargs, ast, env);
    } catch (ConversionException e) {
      throw new EvalException(ast.getLocation(), e.getMessage());
    }
  }

  @SkylarkBuiltin(name = "int", doc = "Creates a rule string class attribute.")
  public static AbstractFunction integer = new AbstractFunction("int") {
      @Override
      public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
          Environment env) throws EvalException, InterruptedException {
        return makeAttr(args, kwargs, "INTEGER", ast, env);
      }
    };

  @SkylarkBuiltin(name = "string", doc = "Creates a rule string class attribute.")
  public static AbstractFunction string = new AbstractFunction("string") {
      @Override
      public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
          Environment env) throws EvalException, InterruptedException {
        return makeAttr(args, kwargs, "STRING", ast, env);
      }
    };

  @SkylarkBuiltin(name = "string_list", doc = "Creates a rule string_list class attribute.")
  public static AbstractFunction stringList = new AbstractFunction("string_list") {
      @Override
      public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
          Environment env) throws EvalException, InterruptedException {
        return makeAttr(args, kwargs, "STRING_LIST", ast, env);
      }
    };

  /**
   * Attr module object exposed in Skylark
   */
  public static final class AttrModule {
    @Override
    public String toString() { return "Attr"; }
    private AttrModule() {}
  }

  // @SkylarkBuiltin(name = "Attr", doc = "Module for creating new attributes.")
  public static final AttrModule module = new AttrModule();

  public static void registerFunctions(Environment env) {
    env.registerFunction(AttrModule.class, integer.getName(), integer);
    env.registerFunction(AttrModule.class, string.getName(), string);
    env.registerFunction(AttrModule.class, stringList.getName(), stringList);
  }

  public static void setupValidationEnvironment(ImmutableMap.Builder<String, Class<?>> builder) {
    builder.put(integer.getName(), Function.class);
    builder.put(integer.getName() + ".return", Object.class);
    builder.put(string.getName(), Function.class);
    builder.put(string.getName() + ".return", Object.class);
    builder.put(stringList.getName(), Function.class);
    builder.put(stringList.getName() + ".return", Object.class);
    builder.put("Attr", Object.class);
  }
}
