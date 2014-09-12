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

import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin.Param;
import com.google.devtools.build.lib.syntax.SkylarkFunction;
import com.google.devtools.build.lib.syntax.SkylarkModule;
import com.google.devtools.build.lib.util.FileTypeSet;

import java.util.List;
import java.util.Map;

/**
 * A helper class to provide Attr module in Skylark.
 */
@SkylarkModule(name = "attr", namespace = true, onlyLoadingPhase = true,
    doc = "Module for creating new attributes.")
public final class SkylarkAttr {
  // TODO(bazel-team): Better check the arguments.

  private static Object makeAttr(Map<String, Object> kwargs, String type,
      FuncallExpression ast, Environment env) throws EvalException {
    try {
      return SkylarkRuleClassFunctions.createAttribute(type, kwargs, ast, env);
    } catch (IllegalStateException | IllegalArgumentException | ConversionException e) {
      throw new EvalException(ast.getLocation(), e.getMessage());
    }
  }

  @SkylarkBuiltin(name = "int", doc = "Creates a rule string class attribute.",
      objectType = SkylarkAttr.class,
      returnType = Attribute.class,
      optionalParams = {
      @Param(name = "default", doc = "the default value of the attribute"),
      @Param(name = "flags", type = List.class, doc = ""),
      @Param(name = "mandatory", type = Boolean.class, doc = ""),
      @Param(name = "cfg", type = String.class, doc = "configuration of the attribute")})
  private static SkylarkFunction integer = new SkylarkFunction("int") {
      @Override
      public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException {
        return makeAttr(kwargs, "INTEGER", ast, env);
      }
    };

  @SkylarkBuiltin(name = "string", doc = "Creates a rule string class attribute.",
      objectType = SkylarkAttr.class,
      returnType = Attribute.class,
      optionalParams = {
      @Param(name = "default", doc = "the default value of the attribute"),
      @Param(name = "flags", type = List.class, doc = ""),
      @Param(name = "mandatory", type = Boolean.class, doc = ""),
      @Param(name = "cfg", type = String.class, doc = "configuration of the attribute")})
  private static SkylarkFunction string = new SkylarkFunction("string") {
      @Override
      public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException {
        return makeAttr(kwargs, "STRING", ast, env);
      }
    };

  @SkylarkBuiltin(name = "label", doc = "Creates a rule string class attribute.",
      objectType = SkylarkAttr.class,
      returnType = Attribute.class,
      optionalParams = {
      @Param(name = "default", doc = "the default value of the attribute"),
      @Param(name = "flags", type = List.class, doc = ""),
      @Param(name = "file_types", type = FileTypeSet.class,
          doc = "allowed file types of the label type attribute"),
      @Param(name = "mandatory", type = Boolean.class, doc = ""),
      @Param(name = "rule_classes", doc = "allowed rule classes of the label type attribute"),
      @Param(name = "cfg", type = String.class, doc = "configuration of the attribute")})
  private static SkylarkFunction label = new SkylarkFunction("label") {
      @Override
      public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException {
        return makeAttr(kwargs, "LABEL", ast, env);
      }
    };

  @SkylarkBuiltin(name = "string_list", doc = "Creates a rule string_list class attribute.",
      objectType = SkylarkAttr.class,
      returnType = Attribute.class,
      optionalParams = {
      @Param(name = "default", doc = "the default value of the attribute"),
      @Param(name = "flags", type = List.class, doc = ""),
      @Param(name = "mandatory", type = Boolean.class, doc = ""),
      @Param(name = "cfg", type = String.class, doc = "configuration of the attribute")})
  private static SkylarkFunction stringList = new SkylarkFunction("string_list") {
      @Override
      public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException {
        return makeAttr(kwargs, "STRING_LIST", ast, env);
      }
    };

  @SkylarkBuiltin(name = "label_list", doc = "Creates a rule label_list class attribute.",
      objectType = SkylarkAttr.class,
      returnType = Attribute.class,
      optionalParams = {
      @Param(name = "default", doc = "the default value of the attribute"),
      @Param(name = "flags", type = List.class, doc = ""),
      @Param(name = "file_types", type = FileTypeSet.class,
          doc = "allowed file types of the label type attribute"),
      @Param(name = "mandatory", type = Boolean.class, doc = ""),
      @Param(name = "rule_classes", doc = "allowed rule classes of the label type attribute"),
      @Param(name = "cfg", type = String.class, doc = "configuration of the attribute")})
  private static SkylarkFunction labelList = new SkylarkFunction("label_list") {
      @Override
      public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException {
        return makeAttr(kwargs, "LABEL_LIST", ast, env);
      }
    };

  @SkylarkBuiltin(name = "bool", doc = "Creates a rule bool class attribute.",
      objectType = SkylarkAttr.class,
      returnType = Attribute.class,
      optionalParams = {
      @Param(name = "default", doc = "the default value of the attribute"),
      @Param(name = "flags", type = List.class, doc = ""),
      @Param(name = "mandatory", type = Boolean.class, doc = ""),
      @Param(name = "cfg", type = String.class, doc = "configuration of the attribute")})
  private static SkylarkFunction bool = new SkylarkFunction("bool") {
      @Override
      public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException {
        return makeAttr(kwargs, "BOOLEAN", ast, env);
      }
    };

  @SkylarkBuiltin(name = "output_list", doc = "Creates a rule output_list class attribute.",
      objectType = SkylarkAttr.class,
      returnType = Attribute.class,
      optionalParams = {
      @Param(name = "default", doc = "the default value of the attribute"),
      @Param(name = "flags", type = List.class, doc = ""),
      @Param(name = "mandatory", type = Boolean.class, doc = ""),
      @Param(name = "cfg", type = String.class, doc = "configuration of the attribute")})
  private static SkylarkFunction outputList = new SkylarkFunction("output_list") {
      @Override
      public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException {
        return makeAttr(kwargs, "OUTPUT_LIST", ast, env);
      }
    };

  @SkylarkBuiltin(name = "license", doc = "Creates a rule license class attribute.",
      objectType = SkylarkAttr.class,
      returnType = Attribute.class,
      optionalParams = {
      @Param(name = "default", doc = "the default value of the attribute"),
      @Param(name = "flags", type = List.class, doc = ""),
      @Param(name = "mandatory", type = Boolean.class, doc = ""),
      @Param(name = "cfg", type = String.class, doc = "configuration of the attribute")})
  private static SkylarkFunction license = new SkylarkFunction("license") {
      @Override
      public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException {
        return makeAttr(kwargs, "LICENSE", ast, env);
      }
    };
}
