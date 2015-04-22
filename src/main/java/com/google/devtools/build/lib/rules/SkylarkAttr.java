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

import static com.google.devtools.build.lib.syntax.SkylarkType.castList;

import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.Attribute.SkylarkLateBound;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.syntax.BuiltinFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.SkylarkCallbackFunction;
import com.google.devtools.build.lib.syntax.SkylarkEnvironment;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkModule;
import com.google.devtools.build.lib.syntax.SkylarkSignature;
import com.google.devtools.build.lib.syntax.SkylarkSignature.Param;
import com.google.devtools.build.lib.syntax.SkylarkSignatureProcessor;
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

  private static final String NON_EMPTY_DOC =
      "set to True if the attribute must not be empty";

  private static final String ALLOW_FILES_DOC =
      "whether File targets are allowed. Can be True, False (default), or "
      + "a FileType filter.";

  private static final String ALLOW_RULES_DOC =
      "which rule targets (name of the classes) are allowed. This is deprecated (kept only for "
      + "compatiblity), use providers instead.";

  private static final String FLAGS_DOC =
      "deprecated, will be removed";

  private static final String DEFAULT_DOC =
      "sets the default value of the attribute.";

  private static final String CONFIGURATION_DOC =
      "configuration of the attribute. "
      + "For example, use DATA_CFG or HOST_CFG.";

  private static final String EXECUTABLE_DOC =
      "set to True if the labels have to be executable. This means the label must refer to an "
      + "executable file, or to a rule that outputs an executable file. Access the labels with "
      + "<code>ctx.executable.&lt;attribute_name&gt;</code>.";

  private static boolean containsNonNoneKey(Map<String, Object> arguments, String key) {
    return arguments.containsKey(key) && arguments.get(key) != Environment.NONE;
  }

  private static Attribute.Builder<?> createAttribute(Type<?> type, Map<String, Object> arguments,
      FuncallExpression ast, SkylarkEnvironment env) throws EvalException, ConversionException {
    // We use an empty name now so that we can set it later.
    // This trick makes sense only in the context of Skylark (builtin rules should not use it).
    Attribute.Builder<?> builder = Attribute.attr("", type);

    Object defaultValue = arguments.get("default");
    if (!EvalUtils.isNullOrNone(defaultValue)) {
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

    if (containsNonNoneKey(arguments, "mandatory") && (Boolean) arguments.get("mandatory")) {
      builder.setPropertyFlag("MANDATORY");
    }

    if (containsNonNoneKey(arguments, "non_empty") && (Boolean) arguments.get("non_empty")) {
      builder.setPropertyFlag("NON_EMPTY");
    }

    if (containsNonNoneKey(arguments, "executable") && (Boolean) arguments.get("executable")) {
      builder.setPropertyFlag("EXECUTABLE");
    }

    if (containsNonNoneKey(arguments, "single_file") && (Boolean) arguments.get("single_file")) {
      builder.setPropertyFlag("SINGLE_ARTIFACT");
    }

    if (containsNonNoneKey(arguments, "allow_files")) {
      Object fileTypesObj = arguments.get("allow_files");
      if (fileTypesObj == Boolean.TRUE) {
        builder.allowedFileTypes(FileTypeSet.ANY_FILE);
      } else if (fileTypesObj == Boolean.FALSE) {
        builder.allowedFileTypes(FileTypeSet.NO_FILE);
      } else if (fileTypesObj instanceof SkylarkFileType) {
        builder.allowedFileTypes(((SkylarkFileType) fileTypesObj).getFileTypeSet());
      } else {
        throw new EvalException(ast.getLocation(),
            "allow_files should be a boolean or a filetype object.");
      }
    } else if (type.equals(Type.LABEL) || type.equals(Type.LABEL_LIST)) {
      builder.allowedFileTypes(FileTypeSet.NO_FILE);
    }

    Object ruleClassesObj = arguments.get("allow_rules");
    if (ruleClassesObj != null && ruleClassesObj != Environment.NONE) {
      builder.allowedRuleClasses(castList(ruleClassesObj, String.class,
              "allowed rule classes for attribute definition"));
    }

    if (containsNonNoneKey(arguments, "providers")) {
      builder.mandatoryProviders(castList(arguments.get("providers"), String.class));
    }

    if (containsNonNoneKey(arguments, "cfg")) {
      builder.cfg((ConfigurationTransition) arguments.get("cfg"));
    }
    return builder;
  }

  private static Attribute.Builder<?> createAttribute(Map<String, Object> kwargs, Type<?> type,
      FuncallExpression ast, Environment env) throws EvalException {
    try {
      return createAttribute(type, kwargs, ast, (SkylarkEnvironment) env);
    } catch (ConversionException e) {
      throw new EvalException(ast.getLocation(), e.getMessage());
    }
  }

  @SkylarkSignature(name = "int", doc =
      "Creates an attribute of type int.",
      objectType = SkylarkAttr.class,
      returnType = Attribute.Builder.class,
      optionalNamedOnly = {
        @Param(name = "default", type = Integer.class, defaultValue = "0",
            doc = DEFAULT_DOC + " If not specified, default is 0."),
        @Param(name = "flags", type = SkylarkList.class, generic1 = String.class,
            defaultValue = "[]", doc = FLAGS_DOC),
        @Param(name = "mandatory", type = Boolean.class, defaultValue = "False",
            doc = MANDATORY_DOC),
        @Param(name = "cfg", type = ConfigurationTransition.class, noneable = true,
            defaultValue = "None", doc = CONFIGURATION_DOC)},
      useAst = true, useEnvironment = true)
  private static BuiltinFunction integer = new BuiltinFunction("int") {
      public Attribute.Builder<?> invoke(Integer defaultInt,
          SkylarkList flags, Boolean mandatory, Object cfg,
          FuncallExpression ast, Environment env) throws EvalException {
        return createAttribute(
            EvalUtils.optionMap(
                "default", defaultInt, "flags", flags, "mandatory", mandatory, "cfg", cfg),
            Type.INTEGER, ast, env);
      }
    };

  @SkylarkSignature(name = "string", doc =
      "Creates an attribute of type string.",
      objectType = SkylarkAttr.class,
      returnType = Attribute.Builder.class,
      optionalNamedOnly = {
        @Param(name = "default", type = String.class,
            defaultValue = "''", doc = DEFAULT_DOC + " If not specified, default is \"\"."),
        @Param(name = "flags", type = SkylarkList.class, generic1 = String.class,
            defaultValue = "[]", doc = FLAGS_DOC),
        @Param(name = "mandatory", type = Boolean.class,
            defaultValue = "False", doc = MANDATORY_DOC),
        @Param(name = "cfg", type = ConfigurationTransition.class, noneable = true,
            defaultValue = "None", doc = CONFIGURATION_DOC)},
      useAst = true, useEnvironment = true)
  private static BuiltinFunction string = new BuiltinFunction("string") {
      public Attribute.Builder<?> invoke(String defaultString,
          SkylarkList flags, Boolean mandatory, Object cfg,
          FuncallExpression ast, Environment env) throws EvalException {
        return createAttribute(
            EvalUtils.optionMap(
                "default", defaultString, "flags", flags, "mandatory", mandatory, "cfg", cfg),
            Type.STRING, ast, env);
      }
    };

  @SkylarkSignature(name = "label", doc =
      "Creates an attribute of type Label. "
      + "It is the only way to specify a dependency to another target. "
      + "If you need a dependency that the user cannot overwrite, make the attribute "
      + "private (starts with <code>_</code>).",
      objectType = SkylarkAttr.class,
      returnType = Attribute.Builder.class,
      optionalNamedOnly = {
        @Param(name = "default", type = Label.class, callbackEnabled = true, noneable = true,
            defaultValue = "None",
            doc = DEFAULT_DOC + " If not specified, default is None. "
            + "Use the <code>Label</code> function to specify a default value."),
        @Param(name = "executable", type = Boolean.class, defaultValue = "False",
            doc = EXECUTABLE_DOC),
        @Param(name = "flags", type = SkylarkList.class, generic1 = String.class,
            defaultValue = "[]", doc = FLAGS_DOC),
        @Param(name = "allow_files", defaultValue = "False", doc = ALLOW_FILES_DOC),
        @Param(name = "mandatory", type = Boolean.class, defaultValue = "False",
            doc = MANDATORY_DOC),
        @Param(name = "providers", type = SkylarkList.class, generic1 = String.class,
            defaultValue = "[]", doc = "mandatory providers every dependency has to have"),
        @Param(name = "allow_rules", type = SkylarkList.class, generic1 = String.class,
            noneable = true, defaultValue = "None", doc = ALLOW_RULES_DOC),
        @Param(name = "single_file", type = Boolean.class, defaultValue = "False",
            doc = "if True, the label must correspond to a single File. "
            + "Access it through <code>ctx.file.&lt;attribute_name&gt;</code>."),
        @Param(name = "cfg", type = ConfigurationTransition.class, noneable = true,
            defaultValue = "None", doc = CONFIGURATION_DOC)},
      useAst = true, useEnvironment = true)
  private static BuiltinFunction label = new BuiltinFunction("label") {
      public Attribute.Builder<?> invoke(
          Object defaultO,
          Boolean executable,
          SkylarkList flags,
          Object allowFiles,
          Boolean mandatory,
          SkylarkList providers,
          Object allowRules,
          Boolean singleFile,
          Object cfg,
          FuncallExpression ast, Environment env) throws EvalException {
        return createAttribute(
            EvalUtils.optionMap(
                "default", defaultO, "executable", executable, "flags", flags,
                "allow_files", allowFiles, "mandatory", mandatory, "providers", providers,
                "allow_rules", allowRules, "single_file", singleFile, "cfg", cfg),
            Type.LABEL, ast, env);
      }
    };

  @SkylarkSignature(name = "string_list", doc =
      "Creates an attribute of type list of strings",
      objectType = SkylarkAttr.class,
      returnType = Attribute.Builder.class,
      optionalPositionals = {
        @Param(name = "default", type = SkylarkList.class, generic1 = String.class,
            defaultValue = "[]",
            doc = DEFAULT_DOC + " If not specified, default is []."),
        @Param(name = "flags", type = SkylarkList.class, generic1 = String.class,
            defaultValue = "[]", doc = FLAGS_DOC),
        @Param(name = "mandatory", type = Boolean.class, defaultValue = "False",
            doc = MANDATORY_DOC),
        @Param(name = "non_empty", type = Boolean.class, defaultValue = "False",
            doc = NON_EMPTY_DOC),
        @Param(name = "cfg", type = ConfigurationTransition.class, noneable = true,
            defaultValue = "None", doc = CONFIGURATION_DOC)},
      useAst = true, useEnvironment = true)
  private static BuiltinFunction stringList = new BuiltinFunction("string_list") {
      public Attribute.Builder<?> invoke(
          SkylarkList defaultList,
          SkylarkList flags,
          Boolean mandatory,
          Boolean nonEmpty,
          Object cfg,
          FuncallExpression ast, Environment env) throws EvalException {
        return createAttribute(
            EvalUtils.optionMap(
                "default", defaultList,
                "flags", flags, "mandatory", mandatory, "non_empty", nonEmpty, "cfg", cfg),
            Type.STRING_LIST, ast, env);
      }
    };

  @SkylarkSignature(name = "label_list", doc =
        "Creates an attribute of type list of labels. "
      + "See <a href=\"#modules.attr.label\">label</a> for more information.",
      objectType = SkylarkAttr.class,
      returnType = Attribute.Builder.class,
      optionalNamedOnly = {
        @Param(name = "default", type = SkylarkList.class, generic1 = Label.class,
            callbackEnabled = true, defaultValue = "[]",
            doc = DEFAULT_DOC + " If not specified, default is []. "
            + "Use the <code>Label</code> function to specify a default value."),
        @Param(name = "allow_files", // bool or FileType filter
            defaultValue = "False", doc = ALLOW_FILES_DOC),
        @Param(name = "allow_rules", type = SkylarkList.class, generic1 = String.class,
            noneable = true, defaultValue = "None", doc = ALLOW_RULES_DOC),
        @Param(name = "providers", type = SkylarkList.class, generic1 = String.class,
            defaultValue = "[]", doc = "mandatory providers every dependency has to have"),
        @Param(name = "flags", type = SkylarkList.class, generic1 = String.class,
            defaultValue = "[]", doc = FLAGS_DOC),
        @Param(name = "mandatory", type = Boolean.class, defaultValue = "False",
            doc = MANDATORY_DOC),
        @Param(name = "non_empty", type = Boolean.class, defaultValue = "False",
            doc = NON_EMPTY_DOC),
        @Param(name = "cfg", type = ConfigurationTransition.class, noneable = true,
            defaultValue = "None", doc = CONFIGURATION_DOC)},
      useAst = true, useEnvironment = true)
  private static BuiltinFunction labelList = new BuiltinFunction("label_list") {
      public Attribute.Builder<?> invoke(
          Object defaultList,
          Object allowFiles,
          Object allowRules,
          SkylarkList providers,
          SkylarkList flags,
          Boolean mandatory,
          Boolean nonEmpty,
          Object cfg,
          FuncallExpression ast, Environment env) throws EvalException {
        return createAttribute(
            EvalUtils.optionMap("default", defaultList,
                "allow_files", allowFiles, "allow_rules", allowRules, "providers", providers,
                "flags", flags, "mandatory", mandatory, "non_empty", nonEmpty, "cfg", cfg),
            Type.LABEL_LIST, ast, env);
      }
    };

  @SkylarkSignature(name = "bool", doc =
      "Creates an attribute of type bool. Its default value is False.",
      objectType = SkylarkAttr.class,
      returnType = Attribute.Builder.class,
      optionalNamedOnly = {
        @Param(name = "default", type = Boolean.class,
            defaultValue = "False", doc = DEFAULT_DOC),
        @Param(name = "flags", type = SkylarkList.class, generic1 = String.class,
            defaultValue = "[]", doc = FLAGS_DOC),
        @Param(name = "mandatory", type = Boolean.class, defaultValue = "False",
            doc = MANDATORY_DOC),
        @Param(name = "cfg", type = ConfigurationTransition.class, noneable = true,
            defaultValue = "None", doc = CONFIGURATION_DOC)},
      useAst = true, useEnvironment = true)
  private static BuiltinFunction bool = new BuiltinFunction("bool") {
      public Attribute.Builder<?> invoke(Boolean defaultBool,
          SkylarkList flags, Boolean mandatory, Object cfg,
          FuncallExpression ast, Environment env) throws EvalException {
        return createAttribute(
            EvalUtils.optionMap(
                "default", defaultBool, "flags", flags, "mandatory", mandatory, "cfg", cfg),
            Type.BOOLEAN, ast, env);
      }
    };

  @SkylarkSignature(name = "output", doc =
        "Creates an attribute of type output. Its default value is None. "
      + "The user provides a file name (string) and the rule must create an action that "
      + "generates the file.",
      objectType = SkylarkAttr.class,
      returnType = Attribute.Builder.class,
      optionalNamedOnly = {
        @Param(name = "default", type = Label.class, noneable = true,
            defaultValue = "None", doc = DEFAULT_DOC),
        @Param(name = "flags", type = SkylarkList.class, generic1 = String.class,
            defaultValue = "[]", doc = FLAGS_DOC),
        @Param(name = "mandatory", type = Boolean.class, defaultValue = "False",
            doc = MANDATORY_DOC),
        @Param(name = "cfg", type = ConfigurationTransition.class, noneable = true,
            defaultValue = "None", doc = CONFIGURATION_DOC)},
      useAst = true, useEnvironment = true)
  private static BuiltinFunction output = new BuiltinFunction("output") {
      public Attribute.Builder<?> invoke(Object defaultO,
          SkylarkList flags, Boolean mandatory, Object cfg,
          FuncallExpression ast, Environment env) throws EvalException {
        return createAttribute(
            EvalUtils.optionMap(
                "default", defaultO, "flags", flags, "mandatory", mandatory, "cfg", cfg),
            Type.OUTPUT, ast, env);
      }
    };

  @SkylarkSignature(name = "output_list", doc =
        "Creates an attribute of type list of outputs. Its default value is <code>[]</code>. "
      + "See <a href=\"#modules.attr.output\">output</a> above for more information.",
      objectType = SkylarkAttr.class,
      returnType = Attribute.Builder.class,
      optionalNamedOnly = {
        @Param(name = "default", type = SkylarkList.class, generic1 = Label.class,
            defaultValue = "[]", doc = DEFAULT_DOC),
        @Param(name = "flags", type = SkylarkList.class, generic1 = String.class,
            defaultValue = "[]", doc = FLAGS_DOC),
        @Param(name = "mandatory", type = Boolean.class,
            defaultValue = "False", doc = MANDATORY_DOC),
        @Param(name = "non_empty", type = Boolean.class, defaultValue = "False",
            doc = NON_EMPTY_DOC),
        @Param(name = "cfg", type = ConfigurationTransition.class, noneable = true,
            defaultValue = "None", doc = CONFIGURATION_DOC)},
      useAst = true, useEnvironment = true)
  private static BuiltinFunction outputList = new BuiltinFunction("output_list") {
      public Attribute.Builder<?> invoke(SkylarkList defaultList,
          SkylarkList flags, Boolean mandatory, Boolean nonEmpty, Object cfg,
          FuncallExpression ast, Environment env) throws EvalException {
        return createAttribute(
            EvalUtils.optionMap("default", defaultList,
                "flags", flags, "mandatory", mandatory, "non_empty", nonEmpty, "cfg", cfg),
            Type.OUTPUT_LIST, ast, env);
      }
    };

  @SkylarkSignature(name = "string_dict", doc =
      "Creates an attribute of type dictionary, mapping from string to string. "
      + "Its default value is dict().",
      objectType = SkylarkAttr.class,
      returnType = Attribute.Builder.class,
      optionalNamedOnly = {
        @Param(name = "default", type = Map.class,
            defaultValue = "{}", doc = DEFAULT_DOC),
        @Param(name = "flags", type = SkylarkList.class, generic1 = String.class,
            defaultValue = "[]", doc = FLAGS_DOC),
        @Param(name = "mandatory", type = Boolean.class, defaultValue = "False",
            doc = MANDATORY_DOC),
        @Param(name = "non_empty", type = Boolean.class, defaultValue = "False",
            doc = NON_EMPTY_DOC),
        @Param(name = "cfg", type = ConfigurationTransition.class, noneable = true,
            defaultValue = "None", doc = CONFIGURATION_DOC)},
      useAst = true, useEnvironment = true)
  private static BuiltinFunction stringDict = new BuiltinFunction("string_dict") {
      public Attribute.Builder<?> invoke(Map<?, ?> defaultO,
          SkylarkList flags, Boolean mandatory, Boolean nonEmpty, Object cfg,
          FuncallExpression ast, Environment env) throws EvalException {
        return createAttribute(
            EvalUtils.optionMap("default", defaultO,
                "flags", flags, "mandatory", mandatory, "non_empty", nonEmpty, "cfg", cfg),
            Type.STRING_DICT, ast, env);
      }
    };

  @SkylarkSignature(name = "license", doc =
      "Creates an attribute of type license. Its default value is NO_LICENSE.",
      // TODO(bazel-team): Implement proper license support for Skylark.
      objectType = SkylarkAttr.class,
      returnType = Attribute.Builder.class,
      optionalNamedOnly = {
        // TODO(bazel-team): ensure this is the correct default value
        @Param(name = "default", defaultValue = "None", noneable = true,
            doc = DEFAULT_DOC),
        @Param(name = "flags", type = SkylarkList.class, generic1 = String.class,
            defaultValue = "[]", doc = FLAGS_DOC),
        @Param(name = "mandatory", type = Boolean.class, defaultValue = "False",
            doc = MANDATORY_DOC),
        @Param(name = "cfg", type = ConfigurationTransition.class, noneable = true,
            defaultValue = "None", doc = CONFIGURATION_DOC)},
      useAst = true, useEnvironment = true)
  private static BuiltinFunction license = new BuiltinFunction("license") {
      public Attribute.Builder<?> invoke(Object defaultO,
          SkylarkList flags, Boolean mandatory, Object cfg,
          FuncallExpression ast, Environment env) throws EvalException {
        return createAttribute(
            EvalUtils.optionMap(
                "default", defaultO, "flags", flags, "mandatory", mandatory, "cfg", cfg),
            Type.LICENSE, ast, env);
      }
    };

  static {
    SkylarkSignatureProcessor.configureSkylarkFunctions(SkylarkAttr.class);
  }
}
