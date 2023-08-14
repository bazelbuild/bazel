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

package com.google.devtools.build.lib.starlarkbuildapi;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;

/**
 * The "attr" module of the Build API.
 *
 * <p>It exposes functions (for example, 'attr.string', 'attr.label_list', etc.) to Starlark users
 * for creating attribute definitions.
 */
@StarlarkBuiltin(
    name = "attr",
    category = DocCategory.TOP_LEVEL_MODULE,
    doc =
        "This is a top-level module for defining the attribute schemas of a rule or aspect. Each"
            + " function returns an object representing the schema of a single attribute. These"
            + " objects are used as the values of the <code>attrs</code> dictionary argument of <a"
            + " href=\"../globals/bzl.html#rule\"><code>rule()</code></a> and <a"
            + " href=\"../globals/bzl.html#aspect\"><code>aspect()</code></a>.<p>See the Rules page"
            + " for more on <a href='https://bazel.build/extending/rules#attributes'>defining</a>"
            + " and <a href='https://bazel.build/extending/rules#implementation_function'>using</a>"
            + " attributes.")
public interface StarlarkAttrModuleApi extends StarlarkValue {

  // dependency and output attributes
  String LABEL_PARAGRAPH =
      "<p>This attribute contains unique <a href='../builtins/Label.html'><code>Label</code></a>"
          + " values. If a string is supplied in place of a <code>Label</code>, it will be"
          + " converted using the <a href='../builtins/Label.html#Label'>label constructor</a>. The"
          + " relative parts of the label path, including the (possibly renamed) repository, are"
          + " resolved with respect to the instantiated target's package.";

  // attr.label, attr.label_list, attr.label_keyed_string_dict
  String DEPENDENCY_ATTR_TEXT =
      LABEL_PARAGRAPH
          + "<p>At analysis time (within the rule's implementation function), when retrieving the"
          + " attribute value from <code>ctx.attr</code>, labels are replaced by the corresponding"
          + " <a href='../builtins/Target.html'><code>Target</code></a>s. This allows you to access"
          + " the providers of the current target's dependencies.";

  // attr.output, attr.output_list
  String OUTPUT_ATTR_TEXT =
      LABEL_PARAGRAPH
          + "<p>At analysis time, the corresponding <a"
          + " href='../builtins/File.html'><code>File</code></a> can be retrieved using <a"
          + " href='../builtins/ctx.html#outputs'><code>ctx.outputs</code></a>.";

  String ALLOW_FILES_ARG = "allow_files";
  String ALLOW_FILES_DOC =
      "Whether <code>File</code> targets are allowed. Can be <code>True</code>, <code>False</code> "
          + "(default), or a list of file extensions that are allowed (for example, "
          + "<code>[\".cc\", \".cpp\"]</code>).";

  String ALLOW_RULES_ARG = "allow_rules";
  String ALLOW_RULES_DOC =
      "Which rule targets (name of the classes) are allowed. This is deprecated (kept only for "
          + "compatibility), use providers instead.";

  String ASPECTS_ARG = "aspects";
  String ASPECTS_ARG_DOC =
      "Aspects that should be applied to the dependency or dependencies specified by this "
          + "attribute.";

  String CONFIGURATION_ARG = "cfg";
  // TODO(b/151742236): Update when new Starlark-based configuration framework is implemented.
  String CONFIGURATION_DOC =
      "<a href=\"https://bazel.build/extending/rules#configurations\">"
          + "Configuration</a> of the attribute. It can be either <code>\"exec\"</code>, which "
          + "indicates that the dependency is built for the <code>execution platform</code>, or "
          + "<code>\"target\"</code>, which indicates that the dependency is build for the "
          + "<code>target platform</code>. A typical example of the difference is when building "
          + "mobile apps, where the <code>target platform</code> is <code>Android</code> or "
          + "<code>iOS</code> while the <code>execution platform</code> is <code>Linux</code>, "
          + "<code>macOS</code>, or <code>Windows</code>.";

  String DEFAULT_ARG = "default";
  // A trailing space is required because it's often prepended to other sentences
  String DEFAULT_DOC =
      "A default value to use if no value for this attribute is given when instantiating the rule.";

  String DOC_ARG = "doc";
  String DOC_DOC =
      "A description of the attribute that can be extracted by documentation generating tools.";

  String EXECUTABLE_ARG = "executable";
  String EXECUTABLE_DOC =
      "True if the dependency has to be executable. This means the label must refer to an "
          + "executable file, or to a rule that outputs an executable file. Access the label "
          + "with <code>ctx.executable.&lt;attribute_name&gt;</code>.";

  String FLAGS_ARG = "flags";
  String FLAGS_DOC = "Deprecated, will be removed.";

  String MANDATORY_ARG = "mandatory";
  String MANDATORY_DOC =
      "If true, the value must be specified explicitly (even if it has a <code>default</code>).";

  String ALLOW_EMPTY_ARG = "allow_empty";
  String ALLOW_EMPTY_DOC = "True if the attribute can be empty.";

  String PROVIDERS_ARG = "providers";
  String PROVIDERS_DOC =
      "The providers that must be given by any dependency appearing in this attribute.<p>The format"
          + " of this argument is a list of lists of providers -- <code>*Info</code> objects"
          + " returned by <a href='../globals/bzl.html#provider'><code>provider()</code></a> (or in"
          + " the case of a legacy provider, its string name). The dependency must return ALL"
          + " providers mentioned in at least ONE of the inner lists. As a convenience, this"
          + " argument may also be a single-level list of providers, in which case it is wrapped in"
          + " an outer list with one element. It is NOT required that the rule of the dependency"
          + " advertises those providers in its <code>provides</code> parameter, however, it is"
          + " considered best practice.";

  String ALLOW_SINGLE_FILE_ARG = "allow_single_file";

  String VALUES_ARG = "values";
  String VALUES_DOC =
      "The list of allowed values for the attribute. An error is raised if any other "
          + "value is given.";

  @StarlarkMethod(
      name = "int",
      doc =
          "Creates a schema for an integer attribute. The value must be in the signed 32-bit range."
              + " The corresponding <a href='../builtins/ctx.html#attr'><code>ctx.attr</code></a>"
              + " attribute will be of type <a href='../core/int.html'><code>int</code></a>.",
      parameters = {
        @Param(
            name = DEFAULT_ARG,
            defaultValue = "0",
            doc = DEFAULT_DOC,
            named = true,
            positional = false),
        @Param(
            name = DOC_ARG,
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            defaultValue = "None",
            doc = DOC_DOC,
            named = true,
            positional = false),
        @Param(
            name = MANDATORY_ARG,
            defaultValue = "False",
            doc = MANDATORY_DOC,
            named = true,
            positional = false),
        @Param(
            name = VALUES_ARG,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = StarlarkInt.class)},
            defaultValue = "[]",
            doc = VALUES_DOC,
            named = true,
            positional = false)
      },
      useStarlarkThread = true)
  Descriptor intAttribute(
      StarlarkInt defaultValue,
      Object doc,
      Boolean mandatory,
      Sequence<?> values,
      StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(
      name = "string",
      doc = "Creates a schema for a <a href='../core/string.html#attr'>string</a> attribute.",
      parameters = {
        @Param(
            name = DEFAULT_ARG,
            defaultValue = "''",
            doc = DEFAULT_DOC,
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = NativeComputedDefaultApi.class)
            },
            named = true,
            positional = false),
        @Param(
            name = DOC_ARG,
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            defaultValue = "None",
            doc = DOC_DOC,
            named = true,
            positional = false),
        @Param(
            name = MANDATORY_ARG,
            defaultValue = "False",
            doc = MANDATORY_DOC,
            named = true,
            positional = false),
        @Param(
            name = VALUES_ARG,
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = String.class),
            },
            defaultValue = "[]",
            doc = VALUES_DOC,
            named = true,
            positional = false)
      },
      useStarlarkThread = true)
  Descriptor stringAttribute(
      Object defaultValue, Object doc, Boolean mandatory, Sequence<?> values, StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(
      name = "label",
      doc =
          "<p>Creates a schema for a label attribute. This is a dependency attribute.</p>"
              + DEPENDENCY_ATTR_TEXT
              + "<p>In addition to ordinary source files, this kind of attribute is often used to"
              + " refer to a tool -- for example, a compiler. Such tools are considered to be"
              + " dependencies, just like source files. To avoid requiring users to specify the"
              + " tool's label every time they use the rule in their BUILD files, you can hard-code"
              + " the label of a canonical tool as the <code>default</code> value of this"
              + " attribute. If you also want to prevent users from overriding this default, you"
              + " can make the attribute private by giving it a name that starts with an"
              + " underscore. See the <a"
              + " href='https://bazel.build/extending/rules#private-attributes'>Rules</a> page"
              + " for more information.",
      parameters = {
        @Param(
            name = DEFAULT_ARG,
            allowedTypes = {
              @ParamType(type = Label.class),
              @ParamType(type = String.class),
              @ParamType(type = LateBoundDefaultApi.class),
              @ParamType(type = NativeComputedDefaultApi.class),
              // TODO(adonovan): remove StarlarkFunction. It's undocumented,
              // unused by Google's .bzl files, and likely unused in Bazel.
              // I suspect it is a vestige of a "computed defaults" feature
              // that was never fully exposed to Starlark (or was since
              // withdrawn).
              @ParamType(type = StarlarkFunction.class),
              @ParamType(type = NoneType.class),
            },
            defaultValue = "None",
            named = true,
            positional = false,
            doc =
                DEFAULT_DOC
                    + "Use a string or the <a"
                    + " href=\"../builtins/Label.html#Label\"><code>Label</code></a> function to"
                    + " specify a default value, for example, <code>attr.label(default ="
                    + " \"//a:b\")</code>."),
        @Param(
            name = DOC_ARG,
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            defaultValue = "None",
            doc = DOC_DOC,
            named = true,
            positional = false),
        @Param(
            name = EXECUTABLE_ARG,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = EXECUTABLE_DOC),
        @Param(
            name = ALLOW_FILES_ARG,
            allowedTypes = {
              @ParamType(type = Boolean.class),
              @ParamType(type = Sequence.class, generic1 = String.class),
              @ParamType(type = NoneType.class),
            },
            defaultValue = "None",
            named = true,
            positional = false,
            doc = ALLOW_FILES_DOC),
        @Param(
            name = ALLOW_SINGLE_FILE_ARG,
            defaultValue = "None",
            named = true,
            positional = false,
            doc =
                "This is similar to <code>allow_files</code>, with the restriction that the label "
                    + "must correspond to a single <a href=\"../builtins/File.html\">File</a>. "
                    + "Access it through <code>ctx.file.&lt;attribute_name&gt;</code>."),
        @Param(
            name = MANDATORY_ARG,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = MANDATORY_DOC),
        @Param(
            name = PROVIDERS_ARG,
            defaultValue = "[]",
            named = true,
            positional = false,
            doc = PROVIDERS_DOC),
        @Param(
            name = ALLOW_RULES_ARG,
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = String.class),
              @ParamType(type = NoneType.class),
            },
            defaultValue = "None",
            named = true,
            positional = false,
            doc = ALLOW_RULES_DOC),
        @Param(
            name = CONFIGURATION_ARG,
            defaultValue = "None",
            named = true,
            positional = false,
            doc =
                CONFIGURATION_DOC
                    + " This parameter is required if <code>executable</code> is True "
                    + "to guard against accidentally building host tools in the "
                    + "target configuration. <code>\"target\"</code> has no semantic "
                    + "effect, so don't set it when <code>executable</code> is False "
                    + "unless it really helps clarify your intentions."),
        @Param(
            name = ASPECTS_ARG,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = StarlarkAspectApi.class)},
            defaultValue = "[]",
            named = true,
            positional = false,
            doc = ASPECTS_ARG_DOC),
        @Param(
            name = FLAGS_ARG,
            defaultValue = "unbound",
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            named = true,
            positional = false,
            documented = false,
            doc = FLAGS_DOC)
      },
      useStarlarkThread = true)
  Descriptor labelAttribute(
      Object defaultValue,
      Object doc,
      Boolean executable,
      Object allowFiles,
      Object allowSingleFile,
      Boolean mandatory,
      Sequence<?> providers,
      Object allowRules,
      Object cfg,
      Sequence<?> aspects,
      Object flags, // Sequence<String> expected
      StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(
      name = "string_list",
      doc = "Creates a schema for a list-of-strings attribute.",
      parameters = {
        @Param(name = MANDATORY_ARG, defaultValue = "False", doc = MANDATORY_DOC, named = true),
        @Param(name = ALLOW_EMPTY_ARG, defaultValue = "True", doc = ALLOW_EMPTY_DOC, named = true),
        @Param(
            name = DEFAULT_ARG,
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = String.class),
              @ParamType(type = NativeComputedDefaultApi.class)
            },
            defaultValue = "[]",
            doc = DEFAULT_DOC,
            named = true,
            positional = false),
        @Param(
            name = DOC_ARG,
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            defaultValue = "None",
            doc = DOC_DOC,
            named = true,
            positional = false)
      },
      useStarlarkThread = true)
  Descriptor stringListAttribute(
      Boolean mandatory, Boolean allowEmpty, Object defaultValue, Object doc, StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(
      name = "int_list",
      doc =
          "Creates a schema for a list-of-integers attribute. Each element must be in the signed"
              + " 32-bit range.",
      parameters = {
        @Param(name = MANDATORY_ARG, defaultValue = "False", doc = MANDATORY_DOC, named = true),
        @Param(name = ALLOW_EMPTY_ARG, defaultValue = "True", doc = ALLOW_EMPTY_DOC, named = true),
        @Param(
            name = DEFAULT_ARG,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = StarlarkInt.class)},
            defaultValue = "[]",
            doc = DEFAULT_DOC,
            named = true,
            positional = false),
        @Param(
            name = DOC_ARG,
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            defaultValue = "None",
            doc = DOC_DOC,
            named = true,
            positional = false)
      },
      useStarlarkThread = true)
  Descriptor intListAttribute(
      Boolean mandatory,
      Boolean allowEmpty,
      Sequence<?> defaultValue,
      Object doc,
      StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(
      name = "label_list",
      doc =
          "<p>Creates a schema for a list-of-labels attribute. This is a dependency attribute. "
              + "The corresponding <a href='../builtins/ctx.html#attr'><code>ctx.attr</code></a> "
              + "attribute will be of type <a href='../core/list.html'>list</a> of "
              + "<a href='../builtins/Target.html'><code>Target</code>s</a>.</p>"
              + DEPENDENCY_ATTR_TEXT,
      parameters = {
        @Param(name = ALLOW_EMPTY_ARG, defaultValue = "True", doc = ALLOW_EMPTY_DOC, named = true),
        @Param(
            name = DEFAULT_ARG,
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = Label.class),
              @ParamType(type = StarlarkFunction.class)
            },
            defaultValue = "[]",
            named = true,
            positional = false,
            doc =
                DEFAULT_DOC
                    + "Use strings or the <a"
                    + " href=\"../builtins/Label.html#Label\"><code>Label</code></a> function to"
                    + " specify default values, for example, <code>attr.label_list(default ="
                    + " [\"//a:b\", \"//a:c\"])</code>."),
        @Param(
            name = DOC_ARG,
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            defaultValue = "None",
            doc = DOC_DOC,
            named = true,
            positional = false),
        @Param(
            name = ALLOW_FILES_ARG,
            allowedTypes = {
              @ParamType(type = Boolean.class),
              @ParamType(type = Sequence.class, generic1 = String.class),
              @ParamType(type = NoneType.class),
            },
            defaultValue = "None",
            named = true,
            positional = false,
            doc = ALLOW_FILES_DOC),
        @Param(
            name = ALLOW_RULES_ARG,
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = String.class),
              @ParamType(type = NoneType.class),
            },
            defaultValue = "None",
            named = true,
            positional = false,
            doc = ALLOW_RULES_DOC),
        @Param(
            name = PROVIDERS_ARG,
            defaultValue = "[]",
            named = true,
            positional = false,
            doc = PROVIDERS_DOC),
        @Param(
            name = FLAGS_ARG,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            defaultValue = "[]",
            named = true,
            positional = false,
            doc = FLAGS_DOC),
        @Param(
            name = MANDATORY_ARG,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = MANDATORY_DOC),
        @Param(
            name = CONFIGURATION_ARG,
            defaultValue = "None",
            named = true,
            positional = false,
            doc = CONFIGURATION_DOC),
        @Param(
            name = ASPECTS_ARG,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = StarlarkAspectApi.class)},
            defaultValue = "[]",
            named = true,
            positional = false,
            doc = ASPECTS_ARG_DOC),
      },
      useStarlarkThread = true)
  Descriptor labelListAttribute(
      Boolean allowEmpty,
      Object defaultValue,
      Object doc,
      Object allowFiles,
      Object allowRules,
      Sequence<?> providers,
      Sequence<?> flags,
      Boolean mandatory,
      Object cfg,
      Sequence<?> aspects,
      StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(
      name = "label_keyed_string_dict",
      doc =
          "<p>Creates a schema for an attribute holding a dictionary, where the keys are labels "
              + "and the values are strings. This is a dependency attribute.</p>"
              + DEPENDENCY_ATTR_TEXT,
      parameters = {
        @Param(name = ALLOW_EMPTY_ARG, defaultValue = "True", doc = ALLOW_EMPTY_DOC, named = true),
        @Param(
            name = DEFAULT_ARG,
            allowedTypes = {
              @ParamType(type = Dict.class),
              @ParamType(type = StarlarkFunction.class)
            },
            defaultValue = "{}",
            named = true,
            positional = false,
            doc =
                DEFAULT_DOC
                    + "Use strings or the <a"
                    + " href=\"../builtins/Label.html#Label\"><code>Label</code></a> function to"
                    + " specify default values, for example,"
                    + " <code>attr.label_keyed_string_dict(default = {\"//a:b\": \"value\","
                    + " \"//a:c\": \"string\"})</code>."),
        @Param(
            name = DOC_ARG,
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            defaultValue = "None",
            doc = DOC_DOC,
            named = true,
            positional = false),
        @Param(
            name = ALLOW_FILES_ARG,
            allowedTypes = {
              @ParamType(type = Boolean.class),
              @ParamType(type = Sequence.class, generic1 = String.class),
              @ParamType(type = NoneType.class),
            },
            defaultValue = "None",
            named = true,
            positional = false,
            doc = ALLOW_FILES_DOC),
        @Param(
            name = ALLOW_RULES_ARG,
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = String.class),
              @ParamType(type = NoneType.class),
            },
            defaultValue = "None",
            named = true,
            positional = false,
            doc = ALLOW_RULES_DOC),
        @Param(
            name = PROVIDERS_ARG,
            defaultValue = "[]",
            named = true,
            positional = false,
            doc = PROVIDERS_DOC),
        @Param(
            name = FLAGS_ARG,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            defaultValue = "[]",
            named = true,
            positional = false,
            doc = FLAGS_DOC),
        @Param(
            name = MANDATORY_ARG,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = MANDATORY_DOC),
        @Param(
            name = CONFIGURATION_ARG,
            defaultValue = "None",
            named = true,
            positional = false,
            doc = CONFIGURATION_DOC),
        @Param(
            name = ASPECTS_ARG,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = StarlarkAspectApi.class)},
            defaultValue = "[]",
            named = true,
            positional = false,
            doc = ASPECTS_ARG_DOC)
      },
      useStarlarkThread = true)
  Descriptor labelKeyedStringDictAttribute(
      Boolean allowEmpty,
      Object defaultValue,
      Object doc,
      Object allowFiles,
      Object allowRules,
      Sequence<?> providers,
      Sequence<?> flags,
      Boolean mandatory,
      Object cfg,
      Sequence<?> aspects,
      StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(
      name = "bool",
      doc =
          "Creates a schema for a boolean attribute. The corresponding <a"
              + " href='../builtins/ctx.html#attr'><code>ctx.attr</code></a> attribute will be of"
              + " type <a href='../core/bool.html'><code>bool</code></a>.",
      parameters = {
        @Param(
            name = DEFAULT_ARG,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = DEFAULT_DOC),
        @Param(
            name = DOC_ARG,
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            defaultValue = "None",
            doc = DOC_DOC,
            named = true,
            positional = false),
        @Param(
            name = MANDATORY_ARG,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = MANDATORY_DOC)
      },
      useStarlarkThread = true)
  Descriptor boolAttribute(
      Boolean defaultValue, Object doc, Boolean mandatory, StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(
      name = "output",
      doc = "<p>Creates a schema for an output (label) attribute.</p>" + OUTPUT_ATTR_TEXT,
      parameters = {
        @Param(
            name = DOC_ARG,
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            defaultValue = "None",
            doc = DOC_DOC,
            named = true,
            positional = false),
        @Param(
            name = MANDATORY_ARG,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = MANDATORY_DOC)
      },
      useStarlarkThread = true)
  Descriptor outputAttribute(Object doc, Boolean mandatory, StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(
      name = "output_list",
      doc = "Creates a schema for a list-of-outputs attribute." + OUTPUT_ATTR_TEXT,
      parameters = {
        @Param(
            name = ALLOW_EMPTY_ARG, //
            defaultValue = "True",
            doc = ALLOW_EMPTY_DOC,
            named = true),
        @Param(
            name = DOC_ARG,
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            defaultValue = "None",
            doc = DOC_DOC,
            named = true,
            positional = false),
        @Param(
            name = MANDATORY_ARG,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = MANDATORY_DOC)
      },
      useStarlarkThread = true)
  Descriptor outputListAttribute(
      Boolean allowEmpty, Object doc, Boolean mandatory, StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(
      name = "string_dict",
      doc =
          "Creates a schema for an attribute holding a dictionary, where the keys and values are "
              + "strings.",
      parameters = {
        @Param(
            name = ALLOW_EMPTY_ARG, //
            defaultValue = "True",
            doc = ALLOW_EMPTY_DOC,
            named = true),
        @Param(
            name = DEFAULT_ARG,
            named = true,
            positional = false,
            defaultValue = "{}",
            doc = DEFAULT_DOC),
        @Param(
            name = DOC_ARG,
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            defaultValue = "None",
            doc = DOC_DOC,
            named = true,
            positional = false),
        @Param(
            name = MANDATORY_ARG,
            named = true,
            positional = false,
            defaultValue = "False",
            doc = MANDATORY_DOC)
      },
      useStarlarkThread = true)
  Descriptor stringDictAttribute(
      Boolean allowEmpty,
      Dict<?, ?> defaultValue,
      Object doc,
      Boolean mandatory,
      StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(
      name = "string_list_dict",
      doc =
          "Creates a schema for an attribute holding a dictionary, where the keys are strings and "
              + "the values are lists of strings.",
      parameters = {
        @Param(
            name = ALLOW_EMPTY_ARG, //
            defaultValue = "True",
            doc = ALLOW_EMPTY_DOC,
            named = true),
        @Param(
            name = DEFAULT_ARG,
            defaultValue = "{}",
            named = true,
            positional = false,
            doc = DEFAULT_DOC),
        @Param(
            name = DOC_ARG,
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            defaultValue = "None",
            doc = DOC_DOC,
            named = true,
            positional = false),
        @Param(
            name = MANDATORY_ARG,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = MANDATORY_DOC)
      },
      useStarlarkThread = true)
  Descriptor stringListDictAttribute(
      Boolean allowEmpty,
      Dict<?, ?> defaultValue,
      Object doc,
      Boolean mandatory,
      StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(
      name = "license",
      doc = "Creates a schema for a license attribute.",
      // TODO(bazel-team): Implement proper license support for Starlark.
      parameters = {
        // TODO(bazel-team): ensure this is the correct default value
        @Param(
            name = DEFAULT_ARG,
            defaultValue = "None",
            named = true,
            positional = false,
            doc = DEFAULT_DOC),
        @Param(
            name = DOC_ARG,
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            defaultValue = "None",
            doc = DOC_DOC,
            named = true,
            positional = false),
        @Param(
            name = MANDATORY_ARG,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = MANDATORY_DOC)
      },
      disableWithFlag = BuildLanguageOptions.INCOMPATIBLE_NO_ATTR_LICENSE,
      useStarlarkThread = true)
  Descriptor licenseAttribute(
      Object defaultValue, Object doc, Boolean mandatory, StarlarkThread thread)
      throws EvalException;

  /** An attribute descriptor. */
  @StarlarkBuiltin(
      name = "Attribute",
      category = DocCategory.BUILTIN,
      doc =
          "Representation of a definition of an attribute. Use the <a"
              + " href=\"../toplevel/attr.html\">attr</a> module to create an Attribute. They are"
              + " only for use with a <a href=\"../globals/bzl.html#rule\">rule</a> or an <a"
              + " href=\"../globals/bzl.html#aspect\">aspect</a>.")
  interface Descriptor extends StarlarkValue {}
}
