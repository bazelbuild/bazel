// Copyright 2014 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.AllowedValueSet;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.Attribute.SkylarkLateBound;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.rules.SkylarkRuleClassFunctions.SkylarkAspect;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature.Param;
import com.google.devtools.build.lib.syntax.BuiltinFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkCallbackFunction;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkSignatureProcessor;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.syntax.Type.ConversionException;
import com.google.devtools.build.lib.syntax.UserDefinedFunction;
import com.google.devtools.build.lib.util.FileTypeSet;

import java.util.ArrayList;
import java.util.List;

/**
 * A helper class to provide Attr module in Skylark.
 *
 * <p>It exposes functions (e.g. 'attr.string', 'attr.label_list', etc.) to Skylark
 * users. The functions are executed through reflection. As everywhere in Skylark,
 * arguments are type-checked with the signature and cannot be null.
 */
@SkylarkModule(
  name = "attr",
  namespace = true,
  doc =
      "Module for creating new attributes. "
          + "They are only for use with the <a href=\"globals.html#rule\">rule</a> function."
)
public final class SkylarkAttr {

  // Arguments

  private static final String ALLOW_FILES_ARG = "allow_files";
  private static final String ALLOW_FILES_DOC =
      "whether File targets are allowed. Can be True, False (default), or a FileType filter.";

  private static final String ALLOW_RULES_ARG = "allow_rules";
  private static final String ALLOW_RULES_DOC =
      "which rule targets (name of the classes) are allowed. This is deprecated (kept only for "
          + "compatiblity), use providers instead.";

  private static final String ASPECTS_ARG = "aspects";
  private static final String ASPECT_ARG_DOC =
      "aspects that should be applied to dependencies specified by this attribute";

  private static final String CONFIGURATION_ARG = "cfg";
  private static final String CONFIGURATION_DOC =
      "configuration of the attribute. For example, use DATA_CFG or HOST_CFG.";

  private static final String DEFAULT_ARG = "default";
  private static final String DEFAULT_DOC = "the default value of the attribute.";

  private static final String EXECUTABLE_ARG = "executable";
  private static final String EXECUTABLE_DOC =
      "True if the labels have to be executable. This means the label must refer to an "
          + "executable file, or to a rule that outputs an executable file. Access the labels "
          + "with <code>ctx.executable.&lt;attribute_name&gt;</code>.";

  private static final String FLAGS_ARG = "flags";
  private static final String FLAGS_DOC = "deprecated, will be removed";

  private static final String MANDATORY_ARG = "mandatory";
  private static final String MANDATORY_DOC = "True if the value must be explicitly specified";

  private static final String NON_EMPTY_ARG = "non_empty";
  private static final String NON_EMPTY_DOC = "True if the attribute must not be empty";

  private static final String PROVIDERS_ARG = "providers";
  private static final String PROVIDERS_DOC =
          "mandatory providers list. It should be either a list of providers, or a "
                  + "list of lists of providers. Every dependency should provide ALL providers "
                  + "from at least ONE of these lists. A single list of providers will be "
                  + "automatically converted to a list containing one list of providers.";

  private static final String SINGLE_FILE_ARG = "single_file";

  private static final String VALUES_ARG = "values";
  private static final String VALUES_DOC =
      "the list of allowed values for the attribute. An error is raised if any other "
          + "value is given.";

  private static boolean containsNonNoneKey(SkylarkDict<String, Object> arguments, String key) {
    return arguments.containsKey(key) && arguments.get(key) != Runtime.NONE;
  }

  private static Attribute.Builder<?> createAttribute(
      Type<?> type, SkylarkDict<String, Object> arguments, FuncallExpression ast, Environment env)
      throws EvalException, ConversionException {
    // We use an empty name now so that we can set it later.
    // This trick makes sense only in the context of Skylark (builtin rules should not use it).
    Attribute.Builder<?> builder = Attribute.attr("", type);

    Object defaultValue = arguments.get(DEFAULT_ARG);
    if (!EvalUtils.isNullOrNone(defaultValue)) {
      if (defaultValue instanceof UserDefinedFunction) {
        // Late bound attribute. Non label type attributes already caused a type check error.
        builder.value(
            new SkylarkLateBound(
                new SkylarkCallbackFunction((UserDefinedFunction) defaultValue, ast, env)));
      } else {
        builder.defaultValue(defaultValue);
      }
    }

    for (String flag : SkylarkList.castSkylarkListOrNoneToList(
        arguments.get(FLAGS_ARG), String.class, FLAGS_ARG)) {
      builder.setPropertyFlag(flag);
    }

    if (containsNonNoneKey(arguments, MANDATORY_ARG) && (Boolean) arguments.get(MANDATORY_ARG)) {
      builder.setPropertyFlag("MANDATORY");
    }

    if (containsNonNoneKey(arguments, NON_EMPTY_ARG) && (Boolean) arguments.get(NON_EMPTY_ARG)) {
      builder.setPropertyFlag("NON_EMPTY");
    }

    if (containsNonNoneKey(arguments, EXECUTABLE_ARG) && (Boolean) arguments.get(EXECUTABLE_ARG)) {
      builder.setPropertyFlag("EXECUTABLE");
    }

    if (containsNonNoneKey(arguments, SINGLE_FILE_ARG)
        && (Boolean) arguments.get(SINGLE_FILE_ARG)) {
      builder.setPropertyFlag("SINGLE_ARTIFACT");
    }

    if (containsNonNoneKey(arguments, ALLOW_FILES_ARG)) {
      Object fileTypesObj = arguments.get(ALLOW_FILES_ARG);
      if (fileTypesObj == Boolean.TRUE) {
        builder.allowedFileTypes(FileTypeSet.ANY_FILE);
      } else if (fileTypesObj == Boolean.FALSE) {
        builder.allowedFileTypes(FileTypeSet.NO_FILE);
      } else if (fileTypesObj instanceof SkylarkFileType) {
        builder.allowedFileTypes(((SkylarkFileType) fileTypesObj).getFileTypeSet());
      } else {
        throw new EvalException(
            ast.getLocation(), "allow_files should be a boolean or a filetype object.");
      }
    } else if (type.equals(BuildType.LABEL) || type.equals(BuildType.LABEL_LIST)) {
      builder.allowedFileTypes(FileTypeSet.NO_FILE);
    }

    Object ruleClassesObj = arguments.get(ALLOW_RULES_ARG);
    if (ruleClassesObj != null && ruleClassesObj != Runtime.NONE) {
      builder.allowedRuleClasses(
          SkylarkList.castSkylarkListOrNoneToList(
              ruleClassesObj, String.class, "allowed rule classes for attribute definition"));
    }

    List<Object> values = SkylarkList.castSkylarkListOrNoneToList(
        arguments.get(VALUES_ARG), Object.class, VALUES_ARG);
    if (!Iterables.isEmpty(values)) {
      builder.allowedValues(new AllowedValueSet(values));
    }

    if (containsNonNoneKey(arguments, PROVIDERS_ARG)) {
      Object obj = arguments.get(PROVIDERS_ARG);
      SkylarkType.checkType(obj, SkylarkList.class, PROVIDERS_ARG);
      boolean isSingleListOfStr = true;
      for (Object o : (SkylarkList) obj) {
        isSingleListOfStr = o instanceof String;
        if (!isSingleListOfStr) {
          break;
        }
      }
      if (isSingleListOfStr) {
        builder.mandatoryProviders(((SkylarkList<?>) obj).getContents(String.class, PROVIDERS_ARG));
      } else {
        builder.mandatoryProvidersList(getProvidersList((SkylarkList) obj));
      }
    }

    if (containsNonNoneKey(arguments, CONFIGURATION_ARG)) {
      builder.cfg((ConfigurationTransition) arguments.get(CONFIGURATION_ARG));
    }
    return builder;
  }

  private static List<List<String>> getProvidersList(SkylarkList skylarkList) throws EvalException {
    List<List<String>> providersList = new ArrayList<>();
    String errorMsg = "Illegal argument: element in '%s' is of unexpected type. "
            + "Should be list of string, but got %s. "
            + "Notice: one single list of string as 'providers' is still supported.";
    for (Object o : skylarkList) {
      if (!(o instanceof SkylarkList)) {
        throw new EvalException(null, String.format(errorMsg, PROVIDERS_ARG,
                EvalUtils.getDataTypeName(o, true)));
      }
      for (Object value : (SkylarkList) o) {
        if (!(value instanceof String)) {
          throw new EvalException(null, String.format(errorMsg, PROVIDERS_ARG,
              "list with an element of type "
                      + EvalUtils.getDataTypeNameFromClass(value.getClass())));
        }
      }
      providersList.add(((SkylarkList<?>) o).getContents(String.class, PROVIDERS_ARG));
    }
    return providersList;
  }

  private static Descriptor createAttrDescriptor(
      SkylarkDict<String, Object> kwargs, Type<?> type, FuncallExpression ast, Environment env)
      throws EvalException {
    try {
      return new Descriptor(createAttribute(type, kwargs, ast, env));
    } catch (ConversionException e) {
      throw new EvalException(ast.getLocation(), e.getMessage());
    }
  }

  private static Descriptor createNonconfigurableAttrDescriptor(
      SkylarkDict<String, Object> kwargs,
      Type<?> type,
      String whyNotConfigurable,
      FuncallExpression ast,
      Environment env) throws EvalException {
    try {
      return new Descriptor(
          createAttribute(type, kwargs, ast, env).nonconfigurable(whyNotConfigurable));
    } catch (ConversionException e) {
      throw new EvalException(ast.getLocation(), e.getMessage());
    }
  }

  @SkylarkSignature(
    name = "int",
    doc = "Creates an attribute of type int.",
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    optionalNamedOnly = {
      @Param(
        name = DEFAULT_ARG,
        type = Integer.class,
        defaultValue = "0",
        doc = DEFAULT_DOC
      ),
      @Param(name = MANDATORY_ARG, type = Boolean.class, defaultValue = "False", doc = MANDATORY_DOC
      ),
      @Param(
        name = VALUES_ARG,
        type = SkylarkList.class,
        generic1 = Integer.class,
        defaultValue = "[]",
        doc = VALUES_DOC
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction integer =
      new BuiltinFunction("int") {
        public Descriptor invoke(
            Integer defaultInt,
            Boolean mandatory,
            SkylarkList<?> values,
            FuncallExpression ast,
            Environment env)
            throws EvalException {
          // TODO(bazel-team): Replace literal strings with constants.
          env.checkLoadingPhase("attr.int", ast.getLocation());
          return createAttrDescriptor(
              EvalUtils.<String, Object>optionMap(
                  env, DEFAULT_ARG, defaultInt, MANDATORY_ARG, mandatory, VALUES_ARG, values),
              Type.INTEGER,
              ast,
              env);
        }
      };

  @SkylarkSignature(
    name = "string",
    doc = "Creates an attribute of type <a href=\"string.html\">string</a>.",
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    optionalNamedOnly = {
      @Param(
        name = DEFAULT_ARG,
        type = String.class,
        defaultValue = "''",
        doc = DEFAULT_DOC
      ),
      @Param(name = MANDATORY_ARG, type = Boolean.class, defaultValue = "False", doc = MANDATORY_DOC
      ),
      @Param(
        name = VALUES_ARG,
        type = SkylarkList.class,
        generic1 = String.class,
        defaultValue = "[]",
        doc = VALUES_DOC
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction string =
      new BuiltinFunction("string") {
        public Descriptor invoke(
            String defaultString,
            Boolean mandatory,
            SkylarkList<?> values,
            FuncallExpression ast,
            Environment env)
            throws EvalException {
          env.checkLoadingPhase("attr.string", ast.getLocation());
          return createAttrDescriptor(
              EvalUtils.<String, Object>optionMap(
                  env, DEFAULT_ARG, defaultString, MANDATORY_ARG, mandatory, VALUES_ARG, values),
              Type.STRING,
              ast,
              env);
        }
      };

  @SkylarkSignature(
    name = "label",
    doc =
        "Creates an attribute of type <a href=\"Target.html\">Target</a> which is the target "
            + "referred to by the label. "
            + "It is the only way to specify a dependency to another target. "
            + "If you need a dependency that the user cannot overwrite, "
            + "<a href=\"../rules.html#private-attributes\">make the attribute private</a>.",
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    optionalNamedOnly = {
      @Param(
        name = DEFAULT_ARG,
        type = Label.class,
        callbackEnabled = true,
        noneable = true,
        defaultValue = "None",
        doc =
            DEFAULT_DOC
                + " Use the <a href=\"globals.html#Label\"><code>Label</code></a> function to "
                + "specify a default value ex:</p>"
                + "<code>attr.label(default = Label(\"//a:b\"))</code>"
      ),
      @Param(
        name = EXECUTABLE_ARG,
        type = Boolean.class,
        defaultValue = "False",
        doc = EXECUTABLE_DOC
      ),
      @Param(name = ALLOW_FILES_ARG, defaultValue = "False", doc = ALLOW_FILES_DOC),
      @Param(name = MANDATORY_ARG, type = Boolean.class, defaultValue = "False", doc = MANDATORY_DOC
      ),
      @Param(
        name = PROVIDERS_ARG,
        type = SkylarkList.class,
        defaultValue = "[]",
        doc = PROVIDERS_DOC
      ),
      @Param(
        name = ALLOW_RULES_ARG,
        type = SkylarkList.class,
        generic1 = String.class,
        noneable = true,
        defaultValue = "None",
        doc = ALLOW_RULES_DOC
      ),
      @Param(
        name = SINGLE_FILE_ARG,
        type = Boolean.class,
        defaultValue = "False",
        doc =
            "if True, the label must correspond to a single <a href=\"file.html\">File</a>. "
                + "Access it through <code>ctx.file.&lt;attribute_name&gt;</code>."
      ),
      @Param(
        name = CONFIGURATION_ARG,
        type = ConfigurationTransition.class,
        noneable = true,
        defaultValue = "None",
        doc = CONFIGURATION_DOC
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction label =
      new BuiltinFunction("label") {
        public Descriptor invoke(
            Object defaultO,
            Boolean executable,
            Object allowFiles,
            Boolean mandatory,
            SkylarkList<?> providers,
            Object allowRules,
            Boolean singleFile,
            Object cfg,
            FuncallExpression ast,
            Environment env)
            throws EvalException {
          env.checkLoadingPhase("attr.label", ast.getLocation());
          return createAttrDescriptor(
              EvalUtils.<String, Object>optionMap(
                  env,
                  DEFAULT_ARG,
                  defaultO,
                  EXECUTABLE_ARG,
                  executable,
                  ALLOW_FILES_ARG,
                  allowFiles,
                  MANDATORY_ARG,
                  mandatory,
                  PROVIDERS_ARG,
                  providers,
                  ALLOW_RULES_ARG,
                  allowRules,
                  SINGLE_FILE_ARG,
                  singleFile,
                  CONFIGURATION_ARG,
                  cfg),
              BuildType.LABEL,
              ast,
              env);
        }
      };

  @SkylarkSignature(
    name = "string_list",
    doc = "Creates an attribute which is a <a href=\"list.html\">list</a> of "
        + "<a href=\"string.html\">strings</a>.",
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    optionalPositionals = {
      @Param(
        name = DEFAULT_ARG,
        type = SkylarkList.class,
        generic1 = String.class,
        defaultValue = "[]",
        doc = DEFAULT_DOC
      ),
      @Param(name = MANDATORY_ARG, type = Boolean.class, defaultValue = "False", doc = MANDATORY_DOC
      ),
      @Param(name = NON_EMPTY_ARG, type = Boolean.class, defaultValue = "False", doc = NON_EMPTY_DOC
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction stringList =
      new BuiltinFunction("string_list") {
        public Descriptor invoke(
            SkylarkList<?> defaultList,
            Boolean mandatory,
            Boolean nonEmpty,
            FuncallExpression ast,
            Environment env)
            throws EvalException {
          env.checkLoadingPhase("attr.string_list", ast.getLocation());
          return createAttrDescriptor(
              EvalUtils.<String, Object>optionMap(
                  env,
                  DEFAULT_ARG,
                  defaultList,
                  MANDATORY_ARG,
                  mandatory,
                  NON_EMPTY_ARG,
                  nonEmpty),
              Type.STRING_LIST,
              ast,
              env);
        }
      };

  @SkylarkSignature(
    name = "int_list",
    doc = "Creates an attribute which is a <a href=\"list.html\">list</a> of ints",
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    optionalPositionals = {
      @Param(
        name = DEFAULT_ARG,
        type = SkylarkList.class,
        generic1 = Integer.class,
        defaultValue = "[]",
        doc = DEFAULT_DOC
      ),
      @Param(name = MANDATORY_ARG, type = Boolean.class, defaultValue = "False", doc = MANDATORY_DOC
      ),
      @Param(name = NON_EMPTY_ARG, type = Boolean.class, defaultValue = "False", doc = NON_EMPTY_DOC
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction intList =
      new BuiltinFunction("int_list") {
        public Descriptor invoke(
            SkylarkList<?> defaultList,
            Boolean mandatory,
            Boolean nonEmpty,
            FuncallExpression ast,
            Environment env)
            throws EvalException {
          env.checkLoadingPhase("attr.int_list", ast.getLocation());
          return createAttrDescriptor(
              EvalUtils.<String, Object>optionMap(
                  env,
                  DEFAULT_ARG,
                  defaultList,
                  MANDATORY_ARG,
                  mandatory,
                  NON_EMPTY_ARG,
                  nonEmpty),
              Type.INTEGER_LIST,
              ast,
              env);
        }
      };

  @SkylarkSignature(
    name = "label_list",
    doc =
        "Creates an attribute which is a <a href=\"list.html\">list</a> of type "
            + "<a href=\"Target.html\">Target</a> which are specified by the labels in the list. "
            + "See <a href=\"attr.html#label\">label</a> for more information.",
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    optionalNamedOnly = {
      @Param(
        name = DEFAULT_ARG,
        type = SkylarkList.class,
        generic1 = Label.class,
        callbackEnabled = true,
        defaultValue = "[]",
        doc =
            DEFAULT_DOC
                + " Use the <a href=\"globals.html#Label\"><code>Label</code></a> function to "
                + "specify default values ex:</p>"
                + "<code>attr.label_list(default = [ Label(\"//a:b\"), Label(\"//a:c\") ])</code>"
      ),
      @Param(
        name = ALLOW_FILES_ARG, // bool or FileType filter
        defaultValue = "False",
        doc = ALLOW_FILES_DOC
      ),
      @Param(
        name = ALLOW_RULES_ARG,
        type = SkylarkList.class,
        generic1 = String.class,
        noneable = true,
        defaultValue = "None",
        doc = ALLOW_RULES_DOC
      ),
      @Param(
        name = PROVIDERS_ARG,
        type = SkylarkList.class,
        defaultValue = "[]",
        doc = PROVIDERS_DOC
      ),
      @Param(
        name = FLAGS_ARG,
        type = SkylarkList.class,
        generic1 = String.class,
        defaultValue = "[]",
        doc = FLAGS_DOC
      ),
      @Param(name = MANDATORY_ARG, type = Boolean.class, defaultValue = "False", doc = MANDATORY_DOC
      ),
      @Param(name = NON_EMPTY_ARG, type = Boolean.class, defaultValue = "False", doc = NON_EMPTY_DOC
      ),
      @Param(
        name = CONFIGURATION_ARG,
        type = ConfigurationTransition.class,
        noneable = true,
        defaultValue = "None",
        doc = CONFIGURATION_DOC
      ),
      @Param(
        name = ASPECTS_ARG,
        type = SkylarkList.class,
        generic1 = SkylarkAspect.class,
        defaultValue = "[]",
        doc = ASPECT_ARG_DOC
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction labelList =
      new BuiltinFunction("label_list") {
        public Descriptor invoke(
            Object defaultList,
            Object allowFiles,
            Object allowRules,
            SkylarkList<?> providers,
            SkylarkList<?> flags,
            Boolean mandatory,
            Boolean nonEmpty,
            Object cfg,
            SkylarkList<?> aspects,
            FuncallExpression ast,
            Environment env)
            throws EvalException {
          env.checkLoadingPhase("attr.label_list", ast.getLocation());
          SkylarkDict<String, Object> kwargs = EvalUtils.<String, Object>optionMap(
              env,
              DEFAULT_ARG,
              defaultList,
              ALLOW_FILES_ARG,
              allowFiles,
              ALLOW_RULES_ARG,
              allowRules,
              PROVIDERS_ARG,
              providers,
              FLAGS_ARG,
              flags,
              MANDATORY_ARG,
              mandatory,
              NON_EMPTY_ARG,
              nonEmpty,
              CONFIGURATION_ARG,
              cfg);
          try {
            Attribute.Builder<?> attribute = createAttribute(
                BuildType.LABEL_LIST, kwargs, ast, env);
            ImmutableList<SkylarkAspect> skylarkAspects =
                ImmutableList.copyOf(aspects.getContents(SkylarkAspect.class, "aspects"));
            return new Descriptor(attribute, skylarkAspects);
          } catch (EvalException e) {
            throw new EvalException(ast.getLocation(), e.getMessage(), e);
          }
        }
      };

  @SkylarkSignature(
    name = "bool",
    doc = "Creates an attribute of type bool.",
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    optionalNamedOnly = {
      @Param(name = DEFAULT_ARG, type = Boolean.class, defaultValue = "False", doc = DEFAULT_DOC),
      @Param(name = MANDATORY_ARG, type = Boolean.class, defaultValue = "False", doc = MANDATORY_DOC
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction bool =
      new BuiltinFunction("bool") {
        public Descriptor invoke(
            Boolean defaultO, Boolean mandatory, FuncallExpression ast, Environment env)
            throws EvalException {
          env.checkLoadingPhase("attr.bool", ast.getLocation());
          return createAttrDescriptor(
              EvalUtils.<String, Object>optionMap(
                  env, DEFAULT_ARG, defaultO, MANDATORY_ARG, mandatory),
              Type.BOOLEAN,
              ast,
              env);
        }
      };

  @SkylarkSignature(
    name = "output",
    doc =
        "Creates an attribute of type output. "
            + "The user provides a file name (string) and the rule must create an action that "
            + "generates the file.",
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    optionalNamedOnly = {
      @Param(
        name = DEFAULT_ARG,
        type = Label.class,
        noneable = true,
        defaultValue = "None",
        doc = DEFAULT_DOC
      ),
      @Param(name = MANDATORY_ARG, type = Boolean.class, defaultValue = "False", doc = MANDATORY_DOC
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction output =
      new BuiltinFunction("output") {
        public Descriptor invoke(
            Object defaultO, Boolean mandatory, FuncallExpression ast, Environment env)
            throws EvalException {
          env.checkLoadingPhase("attr.output", ast.getLocation());
          return createNonconfigurableAttrDescriptor(
              EvalUtils.<String, Object>optionMap(
                  env, DEFAULT_ARG, defaultO, MANDATORY_ARG, mandatory),
              BuildType.OUTPUT,
              "output paths are part of the static graph structure",
              ast,
              env);
        }
      };

  @SkylarkSignature(
    name = "output_list",
    doc =
        "Creates an attribute which is a <a href=\"list.html\">list</a> of outputs. "
            + "See <a href=\"attr.html#output\">output</a> for more information.",
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    optionalNamedOnly = {
      @Param(
        name = DEFAULT_ARG,
        type = SkylarkList.class,
        generic1 = Label.class,
        defaultValue = "[]",
        doc = DEFAULT_DOC
      ),
      @Param(name = MANDATORY_ARG, type = Boolean.class, defaultValue = "False", doc = MANDATORY_DOC
      ),
      @Param(name = NON_EMPTY_ARG, type = Boolean.class, defaultValue = "False", doc = NON_EMPTY_DOC
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction outputList =
      new BuiltinFunction("output_list") {
        public Descriptor invoke(
            SkylarkList defaultList,
            Boolean mandatory,
            Boolean nonEmpty,
            FuncallExpression ast,
            Environment env)
            throws EvalException {
          env.checkLoadingPhase("attr.output_list", ast.getLocation());
          return createAttrDescriptor(
              EvalUtils.<String, Object>optionMap(
                  env,
                  DEFAULT_ARG,
                  defaultList,
                  MANDATORY_ARG,
                  mandatory,
                  NON_EMPTY_ARG,
                  nonEmpty),
              BuildType.OUTPUT_LIST,
              ast,
              env);
        }
      };

  @SkylarkSignature(
    name = "string_dict",
    doc = "Creates an attribute of type <a href=\"dict.html\">dict</a>, mapping from "
        + "<a href=\"string.html\">string</a> to <a href=\"string.html\">string</a>.",
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    optionalNamedOnly = {
      @Param(name = DEFAULT_ARG, type = SkylarkDict.class, defaultValue = "{}", doc = DEFAULT_DOC),
      @Param(name = MANDATORY_ARG, type = Boolean.class, defaultValue = "False", doc = MANDATORY_DOC
      ),
      @Param(name = NON_EMPTY_ARG, type = Boolean.class, defaultValue = "False", doc = NON_EMPTY_DOC
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction stringDict =
      new BuiltinFunction("string_dict") {
        public Descriptor invoke(
            SkylarkDict<?, ?> defaultO,
            Boolean mandatory,
            Boolean nonEmpty,
            FuncallExpression ast,
            Environment env)
            throws EvalException {
          env.checkLoadingPhase("attr.string_dict", ast.getLocation());
          return createAttrDescriptor(
              EvalUtils.<String, Object>optionMap(
                  env, DEFAULT_ARG, defaultO, MANDATORY_ARG, mandatory, NON_EMPTY_ARG, nonEmpty),
              Type.STRING_DICT,
              ast,
              env);
        }
      };

  @SkylarkSignature(
    name = "string_list_dict",
    doc = "Creates an attribute of type <a href=\"dict.html\">dict</a>, mapping from "
        + "<a href=\"string.html\">string</a> to <a href=\"list.html\">list</a> of "
        + "<a href=\"string.html\">string</a>.",
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    optionalNamedOnly = {
      @Param(name = DEFAULT_ARG, type = SkylarkDict.class, defaultValue = "{}", doc = DEFAULT_DOC),
      @Param(name = MANDATORY_ARG, type = Boolean.class, defaultValue = "False", doc = MANDATORY_DOC
      ),
      @Param(name = NON_EMPTY_ARG, type = Boolean.class, defaultValue = "False", doc = NON_EMPTY_DOC
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction stringListDict =
      new BuiltinFunction("string_list_dict") {
        public Descriptor invoke(
            SkylarkDict<?, ?> defaultO,
            Boolean mandatory,
            Boolean nonEmpty,
            FuncallExpression ast,
            Environment env)
            throws EvalException {
          env.checkLoadingPhase("attr.string_list_dict", ast.getLocation());
          return createAttrDescriptor(
              EvalUtils.<String, Object>optionMap(
                  env, DEFAULT_ARG, defaultO, MANDATORY_ARG, mandatory, NON_EMPTY_ARG, nonEmpty),
              Type.STRING_LIST_DICT,
              ast,
              env);
        }
      };

  @SkylarkSignature(
    name = "license",
    doc = "Creates an attribute of type license.",
    // TODO(bazel-team): Implement proper license support for Skylark.
    objectType = SkylarkAttr.class,
    returnType = Descriptor.class,
    optionalNamedOnly = {
      // TODO(bazel-team): ensure this is the correct default value
      @Param(name = DEFAULT_ARG, defaultValue = "None", noneable = true, doc = DEFAULT_DOC),
      @Param(name = MANDATORY_ARG, type = Boolean.class, defaultValue = "False", doc = MANDATORY_DOC
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static BuiltinFunction license =
      new BuiltinFunction("license") {
        public Descriptor invoke(
            Object defaultO, Boolean mandatory, FuncallExpression ast, Environment env)
            throws EvalException {
          env.checkLoadingPhase("attr.license", ast.getLocation());
          return createNonconfigurableAttrDescriptor(
              EvalUtils.<String, Object>optionMap(
                  env, DEFAULT_ARG, defaultO, MANDATORY_ARG, mandatory),
              BuildType.LICENSE,
              "loading phase license checking logic assumes non-configurable values",
              ast,
              env);
        }
      };

  /**
   * A descriptor of an attribute defined in Skylark.
   */
  public static final class Descriptor {
    private final Attribute.Builder<?> attributeBuilder;
    private final ImmutableList<SkylarkAspect> aspects;

    public Descriptor(Attribute.Builder<?> attributeBuilder) {
      this(attributeBuilder, ImmutableList.<SkylarkAspect>of());
    }

    public Descriptor(Attribute.Builder<?> attributeBuilder, ImmutableList<SkylarkAspect> aspects) {
      this.attributeBuilder = attributeBuilder;
      this.aspects = aspects;
    }

    public Attribute.Builder<?> getAttributeBuilder() {
      return attributeBuilder;
    }

    public ImmutableList<SkylarkAspect> getAspects() {
      return aspects;
    }
  }

  static {
    SkylarkSignatureProcessor.configureSkylarkFunctions(SkylarkAttr.class);
  }
}
