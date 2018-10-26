// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylarkbuildapi;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkSemantics.FlagIdentifier;
import com.google.devtools.build.lib.syntax.UserDefinedFunction;

/**
 * The "attr" module of the Build API.
 *
 * <p>It exposes functions (for example, 'attr.string', 'attr.label_list', etc.) to Skylark users
 * for creating attribute definitions.
 */
@SkylarkModule(
    name = "attr",
    namespace = true,
    category = SkylarkModuleCategory.BUILTIN,
    doc =
        "This is a top-level module for defining the attribute schemas of a rule or aspect. Each "
            + "function returns an object representing the schema of a single attribute. These "
            + "objects are used as the values of the <code>attrs</code> dictionary argument of "
            + "<a href=\"globals.html#rule\"><code>rule()</code></a> and "
            + "<a href=\"globals.html#aspect\"><code>aspect()</code></a>."
            + ""
            + "<p>See the Rules page for more on "
            + "<a href='../rules.$DOC_EXT#attributes'>defining</a> and "
            + "<a href='../rules.$DOC_EXT#implementation-function'>using</a> attributes.")
public interface SkylarkAttrApi extends SkylarkValue {

  // dependency and output attributes
  static final String LABEL_PARAGRAPH =
      "<p>This attribute contains <a href='Label.html'><code>Label</code></a> values. If a string "
          + "is supplied in place of a <code>Label</code>, it will be converted using the "
          + "<a href='Label.html#Label'>label constructor</a>. The relative parts of the label "
          + "path, including the (possibly renamed) repository, are resolved with respect to the "
          + "instantiated target's package.";

  // attr.label, attr.label_list, attr.label_keyed_string_dict
  static final String DEPENDENCY_ATTR_TEXT =
      LABEL_PARAGRAPH
          + "<p>At analysis time (within the rule's implementation function), when retrieving the "
          + "attribute value from <code>ctx.attr</code>, labels are replaced by the corresponding "
          + "<a href='Target.html'><code>Target</code></a>s. This allows you to access the "
          + "providers of the currrent target's dependencies.";

  // attr.output, attr.output_list
  static final String OUTPUT_ATTR_TEXT =
      LABEL_PARAGRAPH
          + "<p>At analysis time, the corresponding <a href='File.html'><code>File</code></a> can "
          + "be retrieved using <code>ctx.outputs</code>.";

  static final String ALLOW_FILES_ARG = "allow_files";
  static final String ALLOW_FILES_DOC =
      "Whether <code>File</code> targets are allowed. Can be <code>True</code>, <code>False</code> "
          + "(default), or a list of file extensions that are allowed (for example, "
          + "<code>[\".cc\", \".cpp\"]</code>).";

  static final String ALLOW_RULES_ARG = "allow_rules";
  static final String ALLOW_RULES_DOC =
      "Which rule targets (name of the classes) are allowed. This is deprecated (kept only for "
          + "compatibility), use providers instead.";

  static final String ASPECTS_ARG = "aspects";
  static final String ASPECTS_ARG_DOC =
      "Aspects that should be applied to the dependency or dependencies specified by this "
          + "attribute.";

  static final String CONFIGURATION_ARG = "cfg";
  // TODO(bazel-team): Update when new Skylark-based configuration framework is implemented.
  static final String CONFIGURATION_DOC =
      "<a href=\"../rules.$DOC_EXT#configurations\">Configuration</a> of the attribute. It can be "
          + "either <code>\"host\"</code> or <code>\"target\"</code>.";

  static final String DEFAULT_ARG = "default";
  // A trailing space is required because it's often prepended to other sentences
  static final String DEFAULT_DOC =
      "A default value to use if no value for this attribute is given when instantiating the rule.";

  static final String DOC_ARG = "doc";
  static final String DOC_DOC =
      "A description of the attribute that can be extracted by documentation generating tools.";

  static final String EXECUTABLE_ARG = "executable";
  static final String EXECUTABLE_DOC =
      "True if the dependency has to be executable. This means the label must refer to an "
          + "executable file, or to a rule that outputs an executable file. Access the label "
          + "with <code>ctx.executable.&lt;attribute_name&gt;</code>.";

  static final String FLAGS_ARG = "flags";
  static final String FLAGS_DOC = "Deprecated, will be removed.";

  static final String MANDATORY_ARG = "mandatory";
  static final String MANDATORY_DOC =
      "If true, the value must be specified explicitly (even if it has a <code>default</code>).";

  static final String NON_EMPTY_ARG = "non_empty";
  static final String NON_EMPTY_DOC =
      "True if the attribute must not be empty. Deprecated: Use <code>allow_empty</code> instead.";

  static final String ALLOW_EMPTY_ARG = "allow_empty";
  static final String ALLOW_EMPTY_DOC = "True if the attribute can be empty.";

  static final String PROVIDERS_ARG = "providers";
  static final String PROVIDERS_DOC =
      "The providers that must be given by any dependency appearing in this attribute."
          + ""
          + "<p>The format of this argument is a list of lists of providers -- <code>*Info</code> "
          + "objects returned by <a href='globals.html#provider'><code>provider()</code></a> (or "
          + "in the case of a legacy provider, its string name). The dependency must return ALL "
          + "providers mentioned in at least ONE of the inner lists. As a convenience, this "
          + "argument may also be a single-level list of providers, in which case it is wrapped in "
          + "an outer list with one element.";

  static final String SINGLE_FILE_ARG = "single_file";
  static final String ALLOW_SINGLE_FILE_ARG = "allow_single_file";

  static final String VALUES_ARG = "values";
  static final String VALUES_DOC =
      "The list of allowed values for the attribute. An error is raised if any other "
          + "value is given.";

  @SkylarkCallable(
      name = "int",
      doc = "Creates a schema for an integer attribute.",
      parameters = {
        @Param(
            name = DEFAULT_ARG,
            allowedTypes = {
              @ParamType(type = Integer.class),
              @ParamType(type = UserDefinedFunction.class)
            },
            defaultValue = "0",
            doc = DEFAULT_DOC,
            named = true,
            positional = false),
        @Param(
            name = DOC_ARG,
            type = String.class,
            defaultValue = "''",
            doc = DOC_DOC,
            named = true,
            positional = false),
        @Param(
            name = MANDATORY_ARG,
            type = Boolean.class,
            defaultValue = "False",
            doc = MANDATORY_DOC,
            named = true,
            positional = false),
        @Param(
            name = VALUES_ARG,
            type = SkylarkList.class,
            generic1 = Integer.class,
            defaultValue = "[]",
            doc = VALUES_DOC,
            named = true,
            positional = false)
      },
      useAst = true,
      useEnvironment = true)
  Descriptor intAttribute(
      Integer defaultInt,
      String doc,
      Boolean mandatory,
      SkylarkList<?> values,
      FuncallExpression ast,
      Environment env)
      throws EvalException;

  @SkylarkCallable(
      name = "string",
      doc = "Creates a schema for a string attribute.",
      parameters = {
        @Param(
            name = DEFAULT_ARG,
            allowedTypes = {
              @ParamType(type = String.class),
            },
            defaultValue = "''",
            doc = DEFAULT_DOC,
            named = true,
            positional = false),
        @Param(
            name = DOC_ARG,
            type = String.class,
            defaultValue = "''",
            doc = DOC_DOC,
            named = true,
            positional = false),
        @Param(
            name = MANDATORY_ARG,
            type = Boolean.class,
            defaultValue = "False",
            doc = MANDATORY_DOC,
            named = true,
            positional = false),
        @Param(
            name = VALUES_ARG,
            type = SkylarkList.class,
            generic1 = String.class,
            defaultValue = "[]",
            doc = VALUES_DOC,
            named = true,
            positional = false)
      },
      useAst = true,
      useEnvironment = true)
  public Descriptor stringAttribute(
      String defaultString,
      String doc,
      Boolean mandatory,
      SkylarkList<?> values,
      FuncallExpression ast,
      Environment env)
      throws EvalException;

  @SkylarkCallable(
      name = "label",
      doc = "Creates a schema for a label attribute. This is a dependency attribute."
              + DEPENDENCY_ATTR_TEXT
              + "<p>In addition to ordinary source files, this kind of attribute is often used to "
              + "refer to a tool -- for example, a compiler. Such tools are considered to be "
              + "dependencies, just like source files. To avoid requiring users to specify the "
              + "tool's label every time they use the rule in their BUILD files, you can hard-code "
              + "the label of a canonical tool as the <code>default</code> value of this "
              + "attribute. If you also want to prevent users from overriding this default, you "
              + "can make the attribute private by giving it a name that starts with an "
              + "underscore. See the <a href='../rules.$DOC_EXT#private-attributes'>Rules</a> page "
              + "for more information.",
      parameters = {
        @Param(
            name = DEFAULT_ARG,
            allowedTypes = {
              @ParamType(type = Label.class),
              @ParamType(type = String.class),
              @ParamType(type = LateBoundDefaultApi.class),
              @ParamType(type = UserDefinedFunction.class)
            },
            callbackEnabled = true,
            noneable = true,
            defaultValue = "None",
            named = true,
            positional = false,
            doc =
                DEFAULT_DOC
                    + "Use a string or the <a href=\"globals.html#Label\"><code>Label</code></a> "
                    + "function to specify a default value, for example, "
                    + "<code>attr.label(default = \"//a:b\")</code>."),
        @Param(
            name = DOC_ARG,
            type = String.class,
            defaultValue = "''",
            doc = DOC_DOC,
            named = true,
            positional = false),
        @Param(
            name = EXECUTABLE_ARG,
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = EXECUTABLE_DOC),
        @Param(
            name = ALLOW_FILES_ARG,
            defaultValue = "None",
            named = true,
            positional = false,
            noneable = true,
            doc = ALLOW_FILES_DOC),
        @Param(
            name = ALLOW_SINGLE_FILE_ARG,
            defaultValue = "None",
            named = true,
            positional = false,
            noneable = true,
            doc =
                "This is similar to <code>allow_files</code>, with the restriction that the label "
                    + "must correspond to a single <a href=\"File.html\">File</a>. "
                    + "Access it through <code>ctx.file.&lt;attribute_name&gt;</code>."),
        @Param(
            name = MANDATORY_ARG,
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = MANDATORY_DOC),
        @Param(
            name = PROVIDERS_ARG,
            type = SkylarkList.class,
            defaultValue = "[]",
            named = true,
            positional = false,
            doc = PROVIDERS_DOC),
        @Param(
            name = ALLOW_RULES_ARG,
            type = SkylarkList.class,
            generic1 = String.class,
            noneable = true,
            defaultValue = "None",
            named = true,
            positional = false,
            doc = ALLOW_RULES_DOC),
        @Param(
            name = SINGLE_FILE_ARG,
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            positional = false,
            doc =
                "Deprecated: Use <code>allow_single_file</code> instead. "
                    + "If True, the label must correspond to a single "
                    + "<a href=\"File.html\">File</a>. "
                    + "Access it through <code>ctx.file.&lt;attribute_name&gt;</code>."),
        @Param(
            name = CONFIGURATION_ARG,
            type = Object.class,
            noneable = true,
            defaultValue = "None",
            named = true,
            positional = false,
            doc =
                CONFIGURATION_DOC
                    + " This parameter is required if <code>executable</code> is True."),
        @Param(
            name = ASPECTS_ARG,
            type = SkylarkList.class,
            generic1 = SkylarkAspectApi.class,
            defaultValue = "[]",
            named = true,
            positional = false,
            doc = ASPECTS_ARG_DOC),
      },
      useAst = true,
      useEnvironment = true)
  public Descriptor labelAttribute(
      Object defaultO,
      String doc,
      Boolean executable,
      Object allowFiles,
      Object allowSingleFile,
      Boolean mandatory,
      SkylarkList<?> providers,
      Object allowRules,
      Boolean singleFile,
      Object cfg,
      SkylarkList<?> aspects,
      FuncallExpression ast,
      Environment env)
      throws EvalException;

  @SkylarkCallable(
      name = "string_list",
      doc = "Creates a schema for a list-of-strings attribute.",
      parameters = {
        @Param(
            name = MANDATORY_ARG,
            type = Boolean.class,
            defaultValue = "False",
            doc = MANDATORY_DOC,
            named = true),
        @Param(
            name = NON_EMPTY_ARG,
            type = Boolean.class,
            defaultValue = "False",
            doc = NON_EMPTY_DOC,
            named = true),
        @Param(
            name = ALLOW_EMPTY_ARG,
            type = Boolean.class,
            defaultValue = "True",
            doc = ALLOW_EMPTY_DOC,
            named = true),
        @Param(
            name = DEFAULT_ARG,
            allowedTypes = {
              @ParamType(type = SkylarkList.class, generic1 = String.class),
              @ParamType(type = UserDefinedFunction.class)
            },
            defaultValue = "[]",
            doc = DEFAULT_DOC,
            named = true,
            positional = false),
        @Param(
            name = DOC_ARG,
            type = String.class,
            defaultValue = "''",
            doc = DOC_DOC,
            named = true,
            positional = false)
      },
      useAst = true,
      useEnvironment = true)
  public Descriptor stringListAttribute(
      Boolean mandatory,
      Boolean nonEmpty,
      Boolean allowEmpty,
      SkylarkList<?> defaultList,
      String doc,
      FuncallExpression ast,
      Environment env)
      throws EvalException;

  @SkylarkCallable(
      name = "int_list",
      doc = "Creates a schema for a list-of-integers attribute.",
      parameters = {
        @Param(
            name = MANDATORY_ARG,
            type = Boolean.class,
            defaultValue = "False",
            doc = MANDATORY_DOC,
            named = true),
        @Param(
            name = NON_EMPTY_ARG,
            type = Boolean.class,
            defaultValue = "False",
            doc = NON_EMPTY_DOC,
            named = true),
        @Param(
            name = ALLOW_EMPTY_ARG,
            type = Boolean.class,
            defaultValue = "True",
            doc = ALLOW_EMPTY_DOC,
            named = true),
        @Param(
            name = DEFAULT_ARG,
            allowedTypes = {
              @ParamType(type = SkylarkList.class, generic1 = Integer.class),
              @ParamType(type = UserDefinedFunction.class)
            },
            defaultValue = "[]",
            doc = DEFAULT_DOC,
            named = true,
            positional = false),
        @Param(
            name = DOC_ARG,
            type = String.class,
            defaultValue = "''",
            doc = DOC_DOC,
            named = true,
            positional = false)
      },
      useAst = true,
      useEnvironment = true)
  public Descriptor intListAttribute(
      Boolean mandatory,
      Boolean nonEmpty,
      Boolean allowEmpty,
      SkylarkList<?> defaultList,
      String doc,
      FuncallExpression ast,
      Environment env)
      throws EvalException;

  @SkylarkCallable(
      name = "label_list",
      doc =
          "Creates a schema for a list-of-labels attribute. This is a dependency attribute."
              + DEPENDENCY_ATTR_TEXT,
      parameters = {
        @Param(
            name = ALLOW_EMPTY_ARG,
            type = Boolean.class,
            defaultValue = "True",
            doc = ALLOW_EMPTY_DOC,
            named = true),
        @Param(
            name = DEFAULT_ARG,
            allowedTypes = {
              @ParamType(type = SkylarkList.class, generic1 = Label.class),
              @ParamType(type = UserDefinedFunction.class)
            },
            callbackEnabled = true,
            defaultValue = "[]",
            named = true,
            positional = false,
            doc =
                DEFAULT_DOC
                    + "Use strings or the <a href=\"globals.html#Label\"><code>Label</code></a> "
                    + "function to specify default values, for example, "
                    + "<code>attr.label_list(default = [\"//a:b\", \"//a:c\"])</code>."),
        @Param(
            name = DOC_ARG,
            type = String.class,
            defaultValue = "''",
            doc = DOC_DOC,
            named = true,
            positional = false),
        @Param(
            name = ALLOW_FILES_ARG, // bool or FileType filter
            defaultValue = "None",
            named = true,
            positional = false,
            noneable = true,
            doc = ALLOW_FILES_DOC),
        @Param(
            name = ALLOW_RULES_ARG,
            type = SkylarkList.class,
            generic1 = String.class,
            noneable = true,
            defaultValue = "None",
            named = true,
            positional = false,
            doc = ALLOW_RULES_DOC),
        @Param(
            name = PROVIDERS_ARG,
            type = SkylarkList.class,
            defaultValue = "[]",
            named = true,
            positional = false,
            doc = PROVIDERS_DOC),
        @Param(
            name = FLAGS_ARG,
            type = SkylarkList.class,
            generic1 = String.class,
            defaultValue = "[]",
            named = true,
            positional = false,
            doc = FLAGS_DOC),
        @Param(
            name = MANDATORY_ARG,
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = MANDATORY_DOC),
        @Param(
            name = NON_EMPTY_ARG,
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = NON_EMPTY_DOC),
        @Param(
            name = CONFIGURATION_ARG,
            type = Object.class,
            noneable = true,
            defaultValue = "None",
            named = true,
            positional = false,
            doc = CONFIGURATION_DOC),
        @Param(
            name = ASPECTS_ARG,
            type = SkylarkList.class,
            generic1 = SkylarkAspectApi.class,
            defaultValue = "[]",
            named = true,
            positional = false,
            doc = ASPECTS_ARG_DOC),
      },
      useAst = true,
      useEnvironment = true)
  public Descriptor labelListAttribute(
      Boolean allowEmpty,
      Object defaultList,
      String doc,
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
      throws EvalException;

  @SkylarkCallable(
      name = "label_keyed_string_dict",
      doc =
          "Creates a schema for an attribute holding a dictionary, where the keys are labels and "
              + "the values are strings. This is a dependency attribute."
              + DEPENDENCY_ATTR_TEXT,
      parameters = {
        @Param(
            name = ALLOW_EMPTY_ARG,
            type = Boolean.class,
            defaultValue = "True",
            doc = ALLOW_EMPTY_DOC,
            named = true),
        @Param(
            name = DEFAULT_ARG,
            allowedTypes = {
              @ParamType(type = SkylarkDict.class),
              @ParamType(type = UserDefinedFunction.class)
            },
            callbackEnabled = true,
            defaultValue = "{}",
            named = true,
            positional = false,
            doc =
                DEFAULT_DOC
                    + "Use strings or the <a href=\"globals.html#Label\"><code>Label</code></a> "
                    + "function to specify default values, for example, "
                    + "<code>attr.label_keyed_string_dict(default = "
                    + "{\"//a:b\": \"value\", \"//a:c\": \"string\"})</code>."),
        @Param(
            name = DOC_ARG,
            type = String.class,
            defaultValue = "''",
            doc = DOC_DOC,
            named = true,
            positional = false),
        @Param(
            name = ALLOW_FILES_ARG, // bool or FileType filter
            defaultValue = "None",
            named = true,
            positional = false,
            noneable = true,
            doc = ALLOW_FILES_DOC),
        @Param(
            name = ALLOW_RULES_ARG,
            type = SkylarkList.class,
            generic1 = String.class,
            noneable = true,
            defaultValue = "None",
            named = true,
            positional = false,
            doc = ALLOW_RULES_DOC),
        @Param(
            name = PROVIDERS_ARG,
            type = SkylarkList.class,
            defaultValue = "[]",
            named = true,
            positional = false,
            doc = PROVIDERS_DOC),
        @Param(
            name = FLAGS_ARG,
            type = SkylarkList.class,
            generic1 = String.class,
            defaultValue = "[]",
            named = true,
            positional = false,
            doc = FLAGS_DOC),
        @Param(
            name = MANDATORY_ARG,
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = MANDATORY_DOC),
        @Param(
            name = NON_EMPTY_ARG,
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = NON_EMPTY_DOC),
        @Param(
            name = CONFIGURATION_ARG,
            type = Object.class,
            noneable = true,
            defaultValue = "None",
            named = true,
            positional = false,
            doc = CONFIGURATION_DOC),
        @Param(
            name = ASPECTS_ARG,
            type = SkylarkList.class,
            generic1 = SkylarkAspectApi.class,
            defaultValue = "[]",
            named = true,
            positional = false,
            doc = ASPECTS_ARG_DOC)
      },
      useAst = true,
      useEnvironment = true)
  public Descriptor labelKeyedStringDictAttribute(
      Boolean allowEmpty,
      Object defaultList,
      String doc,
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
      throws EvalException;

  @SkylarkCallable(
      name = "bool",
      doc = "Creates a schema for a boolean attribute.",
      parameters = {
        @Param(
            name = DEFAULT_ARG,
            allowedTypes = {
              @ParamType(type = Boolean.class),
              @ParamType(type = UserDefinedFunction.class)
            },
            defaultValue = "False",
            named = true,
            positional = false,
            doc = DEFAULT_DOC),
        @Param(
            name = DOC_ARG,
            type = String.class,
            defaultValue = "''",
            doc = DOC_DOC,
            named = true,
            positional = false),
        @Param(
            name = MANDATORY_ARG,
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = MANDATORY_DOC)
      },
      useAst = true,
      useEnvironment = true)
  public Descriptor boolAttribute(
      Boolean defaultO, String doc, Boolean mandatory, FuncallExpression ast, Environment env)
      throws EvalException;

  @SkylarkCallable(
      name = "output",
      doc =
          "Creates a schema for an output (label) attribute."
              + OUTPUT_ATTR_TEXT,
      parameters = {
        @Param(
            name = DEFAULT_ARG,
            allowedTypes = {
              @ParamType(type = Label.class),
              @ParamType(type = UserDefinedFunction.class)
            },
            noneable = true,
            defaultValue = "None",
            named = true,
            positional = false,
            doc = "<b><code>default<code> is deprecated for <code>attr.output</code>. Use "
                + "Starlark macros to set the default for <code>output<code> and "
                + "<code>output_list</code> attributes. <p>" + DEFAULT_DOC),
        @Param(
            name = DOC_ARG,
            type = String.class,
            defaultValue = "''",
            doc = DOC_DOC,
            named = true,
            positional = false),
        @Param(
            name = MANDATORY_ARG,
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = MANDATORY_DOC)
      },
      useAst = true,
      useEnvironment = true)
  public Descriptor outputAttribute(
      Object defaultO, String doc, Boolean mandatory, FuncallExpression ast, Environment env)
      throws EvalException;

  @SkylarkCallable(
      name = "output_list",
      doc =
          "Creates a schema for a list-of-outputs attribute."
              + OUTPUT_ATTR_TEXT,
      parameters = {
        @Param(
            name = ALLOW_EMPTY_ARG,
            type = Boolean.class,
            defaultValue = "True",
            doc = ALLOW_EMPTY_DOC,
            named = true),
        @Param(
            name = DEFAULT_ARG,
            allowedTypes = {
              @ParamType(type = SkylarkList.class, generic1 = Label.class),
              @ParamType(type = UserDefinedFunction.class)
            },
            noneable = true,
            defaultValue = "None",
            named = true,
            positional = false,
            doc = "<b><code>default<code> is deprecated for <code>attr.output_list</code>. Use "
                + "Starlark macros to set the default for <code>output<code> and "
                + "<code>output_list</code> attributes. <p>" + DEFAULT_DOC),
        @Param(
            name = DOC_ARG,
            type = String.class,
            defaultValue = "''",
            doc = DOC_DOC,
            named = true,
            positional = false),
        @Param(
            name = MANDATORY_ARG,
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = MANDATORY_DOC),
        @Param(
            name = NON_EMPTY_ARG,
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = NON_EMPTY_DOC)
      },
      useAst = true,
      useEnvironment = true)
  public Descriptor outputListAttribute(
      Boolean allowEmpty,
      Object defaultList,
      String doc,
      Boolean mandatory,
      Boolean nonEmpty,
      FuncallExpression ast,
      Environment env)
      throws EvalException;

  @SkylarkCallable(
      name = "string_dict",
      doc =
          "Creates a schema for an attribute holding a dictionary, where the keys and values are "
              + "strings.",
      parameters = {
        @Param(
            name = ALLOW_EMPTY_ARG,
            type = Boolean.class,
            defaultValue = "True",
            doc = ALLOW_EMPTY_DOC,
            named = true),
        @Param(
            name = DEFAULT_ARG,
            allowedTypes = {
              @ParamType(type = SkylarkDict.class),
              @ParamType(type = UserDefinedFunction.class)
            },
            named = true,
            positional = false,
            defaultValue = "{}",
            doc = DEFAULT_DOC),
        @Param(
            name = DOC_ARG,
            type = String.class,
            defaultValue = "''",
            doc = DOC_DOC,
            named = true,
            positional = false),
        @Param(
            name = MANDATORY_ARG,
            type = Boolean.class,
            named = true,
            positional = false,
            defaultValue = "False",
            doc = MANDATORY_DOC),
        @Param(
            name = NON_EMPTY_ARG,
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = NON_EMPTY_DOC)
      },
      useAst = true,
      useEnvironment = true)
  public Descriptor stringDictAttribute(
      Boolean allowEmpty,
      SkylarkDict<?, ?> defaultO,
      String doc,
      Boolean mandatory,
      Boolean nonEmpty,
      FuncallExpression ast,
      Environment env)
      throws EvalException;

  @SkylarkCallable(
      name = "string_list_dict",
      doc =
          "Creates a schema for an attribute holding a dictionary, where the keys are strings and "
              + "the values are lists of strings.",
      parameters = {
        @Param(
            name = ALLOW_EMPTY_ARG,
            type = Boolean.class,
            defaultValue = "True",
            doc = ALLOW_EMPTY_DOC,
            named = true),
        @Param(
            name = DEFAULT_ARG,
            allowedTypes = {
              @ParamType(type = SkylarkDict.class),
              @ParamType(type = UserDefinedFunction.class)
            },
            defaultValue = "{}",
            named = true,
            positional = false,
            doc = DEFAULT_DOC),
        @Param(
            name = DOC_ARG,
            type = String.class,
            defaultValue = "''",
            doc = DOC_DOC,
            named = true,
            positional = false),
        @Param(
            name = MANDATORY_ARG,
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = MANDATORY_DOC),
        @Param(
            name = NON_EMPTY_ARG,
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = NON_EMPTY_DOC)
      },
      useAst = true,
      useEnvironment = true)
  public Descriptor stringListDictAttribute(
      Boolean allowEmpty,
      SkylarkDict<?, ?> defaultO,
      String doc,
      Boolean mandatory,
      Boolean nonEmpty,
      FuncallExpression ast,
      Environment env)
      throws EvalException;

  @SkylarkCallable(
      name = "license",
      doc = "Creates a schema for a license attribute.",
      // TODO(bazel-team): Implement proper license support for Skylark.
      parameters = {
        // TODO(bazel-team): ensure this is the correct default value
        @Param(
            name = DEFAULT_ARG,
            defaultValue = "None",
            noneable = true,
            named = true,
            positional = false,
            doc = DEFAULT_DOC),
        @Param(
            name = DOC_ARG,
            type = String.class,
            defaultValue = "''",
            doc = DOC_DOC,
            named = true,
            positional = false),
        @Param(
            name = MANDATORY_ARG,
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = MANDATORY_DOC)
      },
      disableWithFlag = FlagIdentifier.INCOMPATIBLE_NO_ATTR_LICENSE,
      useAst = true,
      useEnvironment = true)
  public Descriptor licenseAttribute(
      Object defaultO, String doc, Boolean mandatory, FuncallExpression ast, Environment env)
      throws EvalException;

  /** An attribute descriptor. */
  @SkylarkModule(
      name = "Attribute",
      category = SkylarkModuleCategory.NONE,
      doc =
          "Representation of a definition of an attribute. Use the <a href=\"attr.html\">attr</a> "
              + "module to create an Attribute. They are only for use with a "
              + "<a href=\"globals.html#rule\">rule</a> or an "
              + "<a href=\"globals.html#aspect\">aspect</a>.")
  public static interface Descriptor extends SkylarkValue {}
}
