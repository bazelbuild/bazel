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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.syntax.AbstractFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Function;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin.Param;
import com.google.devtools.build.lib.util.FileTypeSet;

import java.util.List;
import java.util.Map;

/**
 * A helper class to provide Attr module in Skylark.
 */
@SkylarkBuiltin(name = "Attr", doc = "A class to create rule attributes.")
public final class SkylarkAttr {
  // TODO(bazel-team): Better check the arguments.

  private static Object makeAttr(List<Object> args, Map<String, Object> kwargs, String type,
      FuncallExpression ast, Environment env) throws EvalException {
    if (args.size() != 1) {
      throw new EvalException(ast.getLocation(), "This function allows only keywords arguments");
    }
    try {
      return SkylarkRuleClassFunctions.createAttribute(type, kwargs, ast, env);
    } catch (IllegalStateException | IllegalArgumentException | ConversionException e) {
      throw new EvalException(ast.getLocation(), e.getMessage());
    }
  }

  @SkylarkBuiltin(name = "int", doc = "Creates a rule string class attribute.",
      optionalParams = {
      @Param(name = "default", doc = "the default value of the attribute"),
      @Param(name = "flags", type = List.class, doc = ""),
      @Param(name = "cfg", type = String.class, doc = "configuration of the attribute")})
  public static AbstractFunction integer = new AbstractFunction("int") {
      @Override
      public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
          Environment env) throws EvalException, InterruptedException {
        return makeAttr(args, kwargs, "INTEGER", ast, env);
      }
    };

  @SkylarkBuiltin(name = "string", doc = "Creates a rule string class attribute.",
      optionalParams = {
      @Param(name = "default", doc = "the default value of the attribute"),
      @Param(name = "flags", type = List.class, doc = ""),
      @Param(name = "cfg", type = String.class, doc = "configuration of the attribute")})
  public static AbstractFunction string = new AbstractFunction("string") {
      @Override
      public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
          Environment env) throws EvalException, InterruptedException {
        return makeAttr(args, kwargs, "STRING", ast, env);
      }
    };

  @SkylarkBuiltin(name = "label", doc = "Creates a rule string class attribute.",
      optionalParams = {
      @Param(name = "default", doc = "the default value of the attribute"),
      @Param(name = "flags", type = List.class, doc = ""),
      @Param(name = "file_types", type = FileTypeSet.class,
          doc = "allowed file types of the label type attribute"),
      @Param(name = "rule_classes", doc = "allowed rule classes of the label type attribute"),
      @Param(name = "cfg", type = String.class, doc = "configuration of the attribute")})
  public static AbstractFunction label = new AbstractFunction("label") {
      @Override
      public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
          Environment env) throws EvalException, InterruptedException {
        return makeAttr(args, kwargs, "LABEL", ast, env);
      }
    };

  @SkylarkBuiltin(name = "string_list", doc = "Creates a rule string_list class attribute.",
      optionalParams = {
      @Param(name = "default", doc = "the default value of the attribute"),
      @Param(name = "flags", type = List.class, doc = ""),
      @Param(name = "cfg", type = String.class, doc = "configuration of the attribute")})
  public static AbstractFunction stringList = new AbstractFunction("string_list") {
      @Override
      public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
          Environment env) throws EvalException, InterruptedException {
        return makeAttr(args, kwargs, "STRING_LIST", ast, env);
      }
    };

  @SkylarkBuiltin(name = "label_list", doc = "Creates a rule label_list class attribute.",
      optionalParams = {
      @Param(name = "default", doc = "the default value of the attribute"),
      @Param(name = "flags", type = List.class, doc = ""),
      @Param(name = "file_types", type = FileTypeSet.class,
          doc = "allowed file types of the label type attribute"),
      @Param(name = "rule_classes", doc = "allowed rule classes of the label type attribute"),
      @Param(name = "cfg", type = String.class, doc = "configuration of the attribute")})
  public static AbstractFunction labelList = new AbstractFunction("label_list") {
      @Override
      public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
          Environment env) throws EvalException, InterruptedException {
        return makeAttr(args, kwargs, "LABEL_LIST", ast, env);
      }
    };

  @SkylarkBuiltin(name = "bool", doc = "Creates a rule bool class attribute.",
      optionalParams = {
      @Param(name = "default", doc = "the default value of the attribute"),
      @Param(name = "flags", type = List.class, doc = ""),
      @Param(name = "cfg", type = String.class, doc = "configuration of the attribute")})
  public static AbstractFunction bool = new AbstractFunction("bool") {
      @Override
      public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
          Environment env) throws EvalException, InterruptedException {
        return makeAttr(args, kwargs, "BOOLEAN", ast, env);
      }
    };

  @SkylarkBuiltin(name = "output_list", doc = "Creates a rule output_list class attribute.",
      optionalParams = {
      @Param(name = "default", doc = "the default value of the attribute"),
      @Param(name = "flags", type = List.class, doc = ""),
      @Param(name = "cfg", type = String.class, doc = "configuration of the attribute")})
  public static AbstractFunction outputList = new AbstractFunction("output_list") {
      @Override
      public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
          Environment env) throws EvalException, InterruptedException {
        return makeAttr(args, kwargs, "OUTPUT_LIST", ast, env);
      }
    };

  @SkylarkBuiltin(name = "license", doc = "Creates a rule license class attribute.",
      optionalParams = {
      @Param(name = "default", doc = "the default value of the attribute"),
      @Param(name = "flags", type = List.class, doc = ""),
      @Param(name = "cfg", type = String.class, doc = "configuration of the attribute")})
  public static AbstractFunction license = new AbstractFunction("license") {
      @Override
      public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
          Environment env) throws EvalException, InterruptedException {
        return makeAttr(args, kwargs, "LICENSE", ast, env);
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

  public static final AttrModule module = new AttrModule();

  private static final ImmutableList<AbstractFunction> attrFunctions = ImmutableList.of(
      integer, string, stringList, label, labelList, bool, outputList, license);

  public static void registerFunctions(Environment env) {
    for (Function fct : attrFunctions) {
      env.registerFunction(AttrModule.class, fct.getName(), fct);
    }
  }

  public static void setupValidationEnvironment(ImmutableMap.Builder<String, Class<?>> builder) {
    builder.put("Attr", Object.class);
  }
}
