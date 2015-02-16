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

import static com.google.devtools.build.lib.syntax.SkylarkFunction.castList;

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.Attribute.SkylarkLateBound;
import com.google.devtools.build.lib.packages.SkylarkFileType;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin.Param;
import com.google.devtools.build.lib.syntax.SkylarkCallbackFunction;
import com.google.devtools.build.lib.syntax.SkylarkEnvironment;
import com.google.devtools.build.lib.syntax.SkylarkFunction;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkModule;
import com.google.devtools.build.lib.syntax.UserDefinedFunction;
import com.google.devtools.build.lib.util.FileTypeSet;

import java.util.Map;

/**
 * A helper class to provide Attr module in Skylark.
 */
@SkylarkModule(name = "attr", namespace = true, onlyLoadingPhase = true,
    doc = "Module for creating new attributes. "
    + "They are only for use with the <code>rule</code> function.")
public final class SkylarkAttr {

  private static final String MANDATORY_DOC =
      "set to True if users have to explicitely specify the value";

  private static final String ALLOW_FILES_DOC =
      "whether File targets are allowed. Can be True, False (default), or "
      + "a FileType filter.";

  private static final String ALLOW_RULES_DOC =
      "which rule targets (name of the classes) are allowed. "
      + "This is deprecated (kept only for compatiblity), use providers instead.";

  private static final String FLAGS_DOC =
      "deprecated, will be removed";

  private static final String DEFAULT_DOC =
      "sets the default value of the attribute.";

  private static final String CONFIGURATION_DOC =
      "configuration of the attribute. "
      + "For example, use DATA_CFG or HOST_CFG.";

  private static final String EXECUTABLE_DOC =
      "set to True if the labels have to be executable. This means the label refers to an "
      + "executable file, or to a rule that outputs an executable file. Access the labels with "
      + "<code>ctx.executable.&lt;attribute_name&gt;</code>.";

  private static Attribute.Builder<?> createAttribute(Type<?> type, Map<String, Object> arguments,
      FuncallExpression ast, SkylarkEnvironment env) throws EvalException, ConversionException {
    final Location loc = ast.getLocation();
    // We use an empty name now so that we can set it later.
    // This trick makes sense only in the context of Skylark (builtin rules should not use it).
    Attribute.Builder<?> builder = Attribute.attr("", type);

    Object defaultValue = arguments.get("default");
    if (defaultValue != null) {
      if (defaultValue instanceof UserDefinedFunction) {
        // Late bound attribute. Non label type attributes already caused a type check error.
        builder.value(new SkylarkLateBound(
            new SkylarkCallbackFunction((UserDefinedFunction) defaultValue, ast, env)));
      } else {
        builder.defaultValue(defaultValue);
      }
    }

    for (String flag : castList(arguments.get("flags"), String.class)) {
      builder.setPropertyFlag(flag);
    }

    if (arguments.containsKey("mandatory") && (Boolean) arguments.get("mandatory")) {
      builder.setPropertyFlag("MANDATORY");
    }

    if (arguments.containsKey("executable") && (Boolean) arguments.get("executable")) {
      builder.setPropertyFlag("EXECUTABLE");
    }

    if (arguments.containsKey("single_file") && (Boolean) arguments.get("single_file")) {
      builder.setPropertyFlag("SINGLE_ARTIFACT");
    }

    if (arguments.containsKey("allow_files")) {
      Object fileTypesObj = arguments.get("allow_files");
      if (fileTypesObj == Boolean.TRUE) {
        builder.allowedFileTypes(FileTypeSet.ANY_FILE);
      } else if (fileTypesObj == Boolean.FALSE) {
        builder.allowedFileTypes(FileTypeSet.NO_FILE);
      } else if (fileTypesObj instanceof SkylarkFileType) {
        builder.allowedFileTypes(((SkylarkFileType) fileTypesObj).getFileTypeSet());
      } else {
        throw new EvalException(loc, "allow_files should be a boolean or a filetype object.");
      }
    } else if (type.equals(Type.LABEL) || type.equals(Type.LABEL_LIST)) {
      builder.allowedFileTypes(FileTypeSet.NO_FILE);
    }

    Object ruleClassesObj = arguments.get("allow_rules");
    if (ruleClassesObj != null) {
      builder.allowedRuleClasses(castList(ruleClassesObj, String.class,
              "allowed rule classes for attribute definition"));
    }

    if (arguments.containsKey("providers")) {
      builder.mandatoryProviders(castList(arguments.get("providers"), String.class));
    }

    if (arguments.containsKey("cfg")) {
      builder.cfg((ConfigurationTransition) arguments.get("cfg"));
    }
    return builder;
  }

  private static Object createAttribute(Map<String, Object> kwargs, Type<?> type,
      FuncallExpression ast, Environment env) throws EvalException {
    try {
      return createAttribute(type, kwargs, ast, (SkylarkEnvironment) env);
    } catch (ConversionException e) {
      throw new EvalException(ast.getLocation(), e.getMessage());
    }
  }

  @SkylarkBuiltin(name = "int", doc =
      "Creates an attribute of type int.",
      objectType = SkylarkAttr.class,
      returnType = Attribute.class,
      optionalParams = {
      @Param(name = "default", type = Integer.class,
          doc = DEFAULT_DOC + " If not specified, default is 0."),
      @Param(name = "flags", type = SkylarkList.class, generic1 = String.class, doc = FLAGS_DOC),
      @Param(name = "mandatory", type = Boolean.class, doc = MANDATORY_DOC),
      @Param(name = "cfg", type = ConfigurationTransition.class, doc = CONFIGURATION_DOC)})
  private static SkylarkFunction integer = new SkylarkFunction("int") {
      @Override
      public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException {
        return createAttribute(kwargs, Type.INTEGER, ast, env);
      }
    };

  @SkylarkBuiltin(name = "string", doc =
      "Creates an attribute of type string.",
      objectType = SkylarkAttr.class,
      returnType = Attribute.class,
      optionalParams = {
      @Param(name = "default", type = String.class,
          doc = DEFAULT_DOC + " If not specified, default is \"\"."),
      @Param(name = "flags", type = SkylarkList.class, generic1 = String.class, doc = FLAGS_DOC),
      @Param(name = "mandatory", type = Boolean.class, doc = MANDATORY_DOC),
      @Param(name = "cfg", type = ConfigurationTransition.class, doc = CONFIGURATION_DOC)})
  private static SkylarkFunction string = new SkylarkFunction("string") {
      @Override
      public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException {
        return createAttribute(kwargs, Type.STRING, ast, env);
      }
    };

  @SkylarkBuiltin(name = "label", doc =
      "Creates an attribute of type Label. "
      + "It is the only way to specify a dependency to another target. "
      + "If you need a dependency that the user cannot overwrite, make the attribute "
      + "private (starts with <code>_</code>).",
      objectType = SkylarkAttr.class,
      returnType = Attribute.class,
      optionalParams = {
      @Param(name = "default", type = Label.class, callbackEnabled = true,
          doc = DEFAULT_DOC + " If not specified, default is None. "
              + "Use the <code>Label</code> function to specify a default value."),
      @Param(name = "executable", type = Boolean.class, doc = EXECUTABLE_DOC),
      @Param(name = "flags", type = SkylarkList.class, generic1 = String.class, doc = FLAGS_DOC),
      @Param(name = "allow_files", doc = ALLOW_FILES_DOC),
      @Param(name = "mandatory", type = Boolean.class, doc = MANDATORY_DOC),
      @Param(name = "providers", type = SkylarkList.class, generic1 = String.class,
          doc = "mandatory providers every dependency has to have"),
      @Param(name = "allow_rules", type = SkylarkList.class, generic1 = String.class,
          doc = ALLOW_RULES_DOC),
      @Param(name = "single_file", doc =
            "if True, the label must correspond to a single File. "
          + "Access it through <code>ctx.file.&lt;attribute_name&gt;</code>."),
      @Param(name = "cfg", type = ConfigurationTransition.class, doc = CONFIGURATION_DOC)})
  private static SkylarkFunction label = new SkylarkFunction("label") {
      @Override
      public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException {
        return createAttribute(kwargs, Type.LABEL, ast, env);
      }
    };

  @SkylarkBuiltin(name = "string_list", doc =
      "Creates an attribute of type list of strings",
      objectType = SkylarkAttr.class,
      returnType = Attribute.class,
      optionalParams = {
      @Param(name = "default", type = SkylarkList.class, generic1 = String.class,
          doc = DEFAULT_DOC + " If not specified, default is []."),
      @Param(name = "flags", type = SkylarkList.class, generic1 = String.class, doc = FLAGS_DOC),
      @Param(name = "mandatory", type = Boolean.class, doc = MANDATORY_DOC),
      @Param(name = "cfg", type = ConfigurationTransition.class,
          doc = CONFIGURATION_DOC)})
  private static SkylarkFunction stringList = new SkylarkFunction("string_list") {
      @Override
      public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException {
        return createAttribute(kwargs, Type.STRING_LIST, ast, env);
      }
    };

  @SkylarkBuiltin(name = "label_list", doc =
      "Creates an attribute of type list of labels. "
      + "See <code>label</code> for more information.",
      objectType = SkylarkAttr.class,
      returnType = Attribute.class,
      optionalParams = {
      @Param(name = "default", type = SkylarkList.class, generic1 = Label.class,
          callbackEnabled = true,
          doc = DEFAULT_DOC + " If not specified, default is []. "
              + "Use the <code>Label</code> function to specify a default value."),
      @Param(name = "flags", type = SkylarkList.class, generic1 = String.class, doc = FLAGS_DOC),
      @Param(name = "allow_files", doc = ALLOW_FILES_DOC),
      @Param(name = "mandatory", type = Boolean.class, doc = MANDATORY_DOC),
      @Param(name = "allow_rules", type = SkylarkList.class, generic1 = String.class,
          doc = ALLOW_RULES_DOC),
      @Param(name = "providers", type = SkylarkList.class, generic1 = String.class,
          doc = "mandatory providers every dependency has to have"),
      @Param(name = "cfg", type = ConfigurationTransition.class, doc = CONFIGURATION_DOC)})
  private static SkylarkFunction labelList = new SkylarkFunction("label_list") {
      @Override
      public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException {
        return createAttribute(kwargs, Type.LABEL_LIST, ast, env);
      }
    };

  @SkylarkBuiltin(name = "bool", doc =
      "Creates an attribute of type bool. Its default value is False.",
      objectType = SkylarkAttr.class,
      returnType = Attribute.class,
      optionalParams = {
      @Param(name = "default", type = Boolean.class, doc = DEFAULT_DOC),
      @Param(name = "flags", type = SkylarkList.class, generic1 = String.class, doc = FLAGS_DOC),
      @Param(name = "mandatory", type = Boolean.class, doc = MANDATORY_DOC),
      @Param(name = "cfg", type = ConfigurationTransition.class, doc = CONFIGURATION_DOC)})
  private static SkylarkFunction bool = new SkylarkFunction("bool") {
      @Override
      public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException {
        return createAttribute(kwargs, Type.BOOLEAN, ast, env);
      }
    };

  @SkylarkBuiltin(name = "output", doc =
      "Creates an attribute of type output. Its default value is None. "
      + "The user provides a file name (string) and the rule must create an action that "
      + "generates the file.",
      objectType = SkylarkAttr.class,
      returnType = Attribute.class,
      optionalParams = {
      @Param(name = "default", type = Label.class, doc = DEFAULT_DOC),
      @Param(name = "flags", type = SkylarkList.class, generic1 = String.class, doc = FLAGS_DOC),
      @Param(name = "mandatory", type = Boolean.class, doc = MANDATORY_DOC),
      @Param(name = "cfg", type = ConfigurationTransition.class, doc = CONFIGURATION_DOC)})
  private static SkylarkFunction output = new SkylarkFunction("output") {
      @Override
      public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException {
        return createAttribute(kwargs, Type.OUTPUT, ast, env);
      }
    };

  @SkylarkBuiltin(name = "output_list", doc =
      "Creates an attribute of type list of outputs. Its default value is []. "
      + "See <code>output</code> above for more information.",
      objectType = SkylarkAttr.class,
      returnType = Attribute.class,
      optionalParams = {
      @Param(name = "default", type = SkylarkList.class, generic1 = Label.class, doc = DEFAULT_DOC),
      @Param(name = "flags", type = SkylarkList.class, generic1 = String.class, doc = FLAGS_DOC),
      @Param(name = "mandatory", type = Boolean.class, doc = MANDATORY_DOC),
      @Param(name = "cfg", type = ConfigurationTransition.class, doc = CONFIGURATION_DOC)})
  private static SkylarkFunction outputList = new SkylarkFunction("output_list") {
      @Override
      public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException {
        return createAttribute(kwargs, Type.OUTPUT_LIST, ast, env);
      }
    };

  @SkylarkBuiltin(name = "string_dict", doc =
      "Creates an attribute of type dictionary, mapping from string to string. "
      + "Its default value is {}.",
      objectType = SkylarkAttr.class,
      returnType = Attribute.class,
      optionalParams = {
      @Param(name = "default", type = Map.class, doc = DEFAULT_DOC),
      @Param(name = "flags", type = SkylarkList.class, generic1 = String.class, doc = FLAGS_DOC),
      @Param(name = "mandatory", type = Boolean.class, doc = MANDATORY_DOC),
      @Param(name = "cfg", type = ConfigurationTransition.class, doc = CONFIGURATION_DOC)})
  private static SkylarkFunction stringDict = new SkylarkFunction("string_dict") {
      @Override
      public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException {
        return createAttribute(kwargs, Type.STRING_DICT, ast, env);
      }
    };

  @SkylarkBuiltin(name = "license", doc =
      "Creates an attribute of type license. Its default value is NO_LICENSE.",
      // TODO(bazel-team): Implement proper license support for Skylark.
      objectType = SkylarkAttr.class,
      returnType = Attribute.class,
      optionalParams = {
      @Param(name = "default", doc = DEFAULT_DOC),
      @Param(name = "flags", type = SkylarkList.class, generic1 = String.class, doc = FLAGS_DOC),
      @Param(name = "mandatory", type = Boolean.class, doc = MANDATORY_DOC),
      @Param(name = "cfg", type = ConfigurationTransition.class, doc = CONFIGURATION_DOC)})
  private static SkylarkFunction license = new SkylarkFunction("license") {
      @Override
      public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException {
        return createAttribute(kwargs, Type.LICENSE, ast, env);
      }
    };
}
