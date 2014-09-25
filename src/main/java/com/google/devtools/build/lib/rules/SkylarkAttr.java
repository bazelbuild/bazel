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
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin.Param;
import com.google.devtools.build.lib.syntax.SkylarkFunction;
import com.google.devtools.build.lib.syntax.SkylarkModule;

import java.util.List;
import java.util.Map;

/**
 * A helper class to provide Attr module in Skylark.
 */
@SkylarkModule(name = "attr", namespace = true, onlyLoadingPhase = true,
    doc = "Module for creating new attributes. "
    + "They are only for use with the <i>rule</i> function.")
public final class SkylarkAttr {
  // TODO(bazel-team): Better check the arguments.

  private static final String MANDATORY_DOC =
      "set to true if users have to explicitely specify the value";

  private static final String FILE_TYPES_DOC =
      "allowed file types of the label type attribute. "
      + "For example, use ANY_FILE, NO_FILE, or the filetype function.";

  private static final String RULE_CLASSES_DOC =
      "allowed rule classes of the label type attribute. "
      + "For example, use ANY_RULE, NO_RULE, or a list of strings.";

  private static final String FLAGS_DOC =
      "deprecated, will be removed";

  private static final String DEFAULT_DOC =
      "the default value of the attribute";

  private static final String CONFIGURATION_DOC =
      "configuration of the attribute. "
      + "For example, use DATA_CFG or HOST_CFG.";

  private static final String EXECUTABLE_DOC =
      "set to true if the labels have to be executable";

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
      @Param(name = "default", doc = DEFAULT_DOC),
      @Param(name = "flags", type = List.class, doc = FLAGS_DOC),
      @Param(name = "mandatory", type = Boolean.class, doc = MANDATORY_DOC),
      @Param(name = "cfg", type = ConfigurationTransition.class, doc = CONFIGURATION_DOC)})
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
      @Param(name = "default", doc = DEFAULT_DOC),
      @Param(name = "flags", type = List.class, doc = FLAGS_DOC),
      @Param(name = "mandatory", type = Boolean.class, doc = MANDATORY_DOC),
      @Param(name = "cfg", type = ConfigurationTransition.class, doc = CONFIGURATION_DOC)})
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
      @Param(name = "default", doc = DEFAULT_DOC),
      @Param(name = "executable", type = Boolean.class, doc = EXECUTABLE_DOC),
      @Param(name = "flags", type = List.class, doc = FLAGS_DOC),
      @Param(name = "file_types", doc = FILE_TYPES_DOC),
      @Param(name = "mandatory", type = Boolean.class, doc = MANDATORY_DOC),
      @Param(name = "rule_classes", doc = RULE_CLASSES_DOC),
      @Param(name = "providers", type = List.class,
          doc = "mandatory providers every dependency has to have"),
      @Param(name = "cfg", type = ConfigurationTransition.class, doc = CONFIGURATION_DOC)})
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
      @Param(name = "default", doc = DEFAULT_DOC),
      @Param(name = "flags", type = List.class, doc = FLAGS_DOC),
      @Param(name = "mandatory", type = Boolean.class, doc = MANDATORY_DOC),
      @Param(name = "cfg", type = ConfigurationTransition.class,
          doc = CONFIGURATION_DOC)})
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
      @Param(name = "default", doc = DEFAULT_DOC),
      @Param(name = "executable", type = Boolean.class, doc = EXECUTABLE_DOC),
      @Param(name = "flags", type = List.class, doc = FLAGS_DOC),
      @Param(name = "file_types", doc = FILE_TYPES_DOC),
      @Param(name = "mandatory", type = Boolean.class, doc = MANDATORY_DOC),
      @Param(name = "rule_classes", doc = RULE_CLASSES_DOC),
      @Param(name = "providers", type = List.class,
          doc = "mandatory providers every dependency has to have"),
      @Param(name = "cfg", type = ConfigurationTransition.class, doc = CONFIGURATION_DOC)})
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
      @Param(name = "default", doc = DEFAULT_DOC),
      @Param(name = "flags", type = List.class, doc = FLAGS_DOC),
      @Param(name = "mandatory", type = Boolean.class, doc = MANDATORY_DOC),
      @Param(name = "cfg", type = ConfigurationTransition.class, doc = CONFIGURATION_DOC)})
  private static SkylarkFunction bool = new SkylarkFunction("bool") {
      @Override
      public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException {
        return makeAttr(kwargs, "BOOLEAN", ast, env);
      }
    };

  @SkylarkBuiltin(name = "output", doc = "Creates a rule output class attribute.",
      objectType = SkylarkAttr.class,
      returnType = Attribute.class,
      optionalParams = {
      @Param(name = "default", doc = DEFAULT_DOC),
      @Param(name = "flags", type = List.class, doc = FLAGS_DOC),
      @Param(name = "mandatory", type = Boolean.class, doc = MANDATORY_DOC),
      @Param(name = "cfg", type = ConfigurationTransition.class, doc = CONFIGURATION_DOC)})
  private static SkylarkFunction output = new SkylarkFunction("output") {
      @Override
      public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException {
        return makeAttr(kwargs, "OUTPUT", ast, env);
      }
    };

  @SkylarkBuiltin(name = "output_list", doc = "Creates a rule output_list class attribute.",
      objectType = SkylarkAttr.class,
      returnType = Attribute.class,
      optionalParams = {
      @Param(name = "default", doc = DEFAULT_DOC),
      @Param(name = "flags", type = List.class, doc = FLAGS_DOC),
      @Param(name = "mandatory", type = Boolean.class, doc = MANDATORY_DOC),
      @Param(name = "cfg", type = ConfigurationTransition.class, doc = CONFIGURATION_DOC)})
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
      @Param(name = "default", doc = DEFAULT_DOC),
      @Param(name = "flags", type = List.class, doc = FLAGS_DOC),
      @Param(name = "mandatory", type = Boolean.class, doc = MANDATORY_DOC),
      @Param(name = "cfg", type = ConfigurationTransition.class, doc = CONFIGURATION_DOC)})
  private static SkylarkFunction license = new SkylarkFunction("license") {
      @Override
      public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException {
        return makeAttr(kwargs, "LICENSE", ast, env);
      }
    };
}
