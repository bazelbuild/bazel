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

import com.google.devtools.build.docgen.annot.DocumentMethods;
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkConfigApi.BuildSettingApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkThread;

/**
 * Interface for a global Starlark library containing rule-related helper and registration
 * functions.
 */
@DocumentMethods
public interface StarlarkRuleFunctionsApi<FileApiT extends FileApi> {

  String EXEC_COMPATIBLE_WITH_PARAM = "exec_compatible_with";
  String TOOLCHAINS_PARAM = "toolchains";

  String PROVIDES_DOC =
      "A list of providers that the implementation function must return."
          + ""
          + "<p>It is an error if the implementation function omits any of the types of providers "
          + "listed here from its return value. However, the implementation function may return "
          + "additional providers not listed here."
          + ""
          + "<p>Each element of the list is an <code>*Info</code> object returned by "
          + "<a href='globals.html#provider'><code>provider()</code></a>, except that a legacy "
          + "provider is represented by its string name instead.";

  @StarlarkMethod(
      name = "provider",
      doc =
          "Creates a declared provider 'constructor'. The return value of this "
              + "function can be used to create \"struct-like\" values. Example:<br>"
              + "<pre class=\"language-python\">data = provider()\n"
              + "d = data(x = 2, y = 3)\n"
              + "print(d.x + d.y) # prints 5</pre>"
              + "<p>See <a href='../rules.$DOC_EXT#providers'>Rules (Providers)</a> for a "
              + "comprehensive guide on how to use providers.",
      parameters = {
        @Param(
            name = "doc",
            named = true,
            defaultValue = "''",
            doc =
                "A description of the provider that can be extracted by documentation generating"
                    + " tools."),
        @Param(
            name = "fields",
            doc =
                "If specified, restricts the set of allowed fields. <br>Possible values are:<ul> "
                    + " <li> list of fields:<br>       <pre"
                    + " class=\"language-python\">provider(fields = ['a', 'b'])</pre><p>  <li>"
                    + " dictionary field name -> documentation:<br>       <pre"
                    + " class=\"language-python\">provider(\n"
                    + "       fields = { 'a' : 'Documentation for a', 'b' : 'Documentation for b'"
                    + " })</pre></ul>All fields are optional.",
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = String.class),
              @ParamType(type = Dict.class),
              @ParamType(type = NoneType.class),
            },
            named = true,
            positional = false,
            defaultValue = "None")
      },
      useStarlarkThread = true)
  ProviderApi provider(String doc, Object fields, StarlarkThread thread) throws EvalException;

  @StarlarkMethod(
      name = "rule",
      doc =
          "Creates a new rule, which can be called from a BUILD file or a macro to create targets."
              + "<p>Rules must be assigned to global variables in a .bzl file; the name of the "
              + "global variable is the rule's name."
              + "<p>Test rules are required to have a name ending in <code>_test</code>, while all "
              + "other rules must not have this suffix. (This restriction applies only to rules, "
              + "not to their targets.)",
      parameters = {
        @Param(
            name = "implementation",
            named = true,
            doc =
                "the Starlark function implementing this rule, must have exactly one parameter: "
                    + "<a href=\"ctx.html\">ctx</a>. The function is called during the analysis "
                    + "phase for each instance of the rule. It can access the attributes "
                    + "provided by the user. It must create actions to generate all the declared "
                    + "outputs."),
        @Param(
            name = "test",
            named = true,
            defaultValue = "False",
            doc =
                "Whether this rule is a test rule, that is, whether it may be the subject of a "
                    + "<code>blaze test</code> command. All test rules are automatically "
                    + "considered <a href='#rule.executable'>executable</a>; it is unnecessary "
                    + "(and discouraged) to explicitly set <code>executable = True</code> for a "
                    + "test rule. See the "
                    + "<a href='../rules.$DOC_EXT#executable-rules-and-test-rules'>Rules page</a> "
                    + "for more information."),
        @Param(
            name = "attrs",
            allowedTypes = {
              @ParamType(type = Dict.class),
              @ParamType(type = NoneType.class),
            },
            named = true,
            defaultValue = "None",
            doc =
                "dictionary to declare all the attributes of the rule. It maps from an attribute "
                    + "name to an attribute object (see <a href=\"attr.html\">attr</a> module). "
                    + "Attributes starting with <code>_</code> are private, and can be used to "
                    + "add an implicit dependency on a label. The attribute <code>name</code> is "
                    + "implicitly added and must not be specified. Attributes "
                    + "<code>visibility</code>, <code>deprecation</code>, <code>tags</code>, "
                    + "<code>testonly</code>, and <code>features</code> are implicitly added and "
                    + "cannot be overridden. Most rules need only a handful of attributes. To "
                    + "limit memory usage, the rule function imposes a cap on the size of attrs."),
        // TODO(bazel-team): need to give the types of these builtin attributes
        @Param(
            name = "outputs",
            allowedTypes = {
              @ParamType(type = Dict.class),
              @ParamType(type = NoneType.class),
              @ParamType(type = StarlarkFunction.class) // a function defined in Starlark
            },
            named = true,
            defaultValue = "None",
            valueWhenDisabled = "None",
            disableWithFlag = BuildLanguageOptions.INCOMPATIBLE_NO_RULE_OUTPUTS_PARAM,
            doc =
                "This parameter has been deprecated. Migrate rules to use"
                    + " <code>OutputGroupInfo</code> or <code>attr.output</code> instead. <p>A"
                    + " schema for defining predeclared outputs. Unlike <a"
                    + " href='attr.html#output'><code>output</code></a> and <a"
                    + " href='attr.html#output_list'><code>output_list</code></a> attributes, the"
                    + " user does not specify the labels for these files. See the <a"
                    + " href='../rules.$DOC_EXT#files'>Rules page</a> for more on predeclared"
                    + " outputs.<p>The value of this argument is either a dictionary or a callback"
                    + " function that produces a dictionary. The callback works similar to"
                    + " computed dependency attributes: The function's parameter names are matched"
                    + " against the rule's attributes, so for example if you pass <code>outputs ="
                    + " _my_func</code> with the definition <code>def _my_func(srcs, deps):"
                    + " ...</code>, the function has access to the attributes <code>srcs</code>"
                    + " and <code>deps</code>. Whether the dictionary is specified directly or via"
                    + " a function, it is interpreted as follows.<p>Each entry in the dictionary"
                    + " creates a predeclared output where the key is an identifier and the value"
                    + " is a string template that determines the output's label. In the rule's"
                    + " implementation function, the identifier becomes the field name used to"
                    + " access the output's <a href='File.html'><code>File</code></a> in <a"
                    + " href='ctx.html#outputs'><code>ctx.outputs</code></a>. The output's label"
                    + " has the same package as the rule, and the part after the package is"
                    + " produced by substituting each placeholder of the form"
                    + " <code>\"%{ATTR}\"</code> with a string formed from the value of the"
                    + " attribute <code>ATTR</code>:<ul><li>String-typed attributes are"
                    + " substituted verbatim.<li>Label-typed attributes become the part of the"
                    + " label after the package, minus the file extension. For example, the label"
                    + " <code>\"//pkg:a/b.c\"</code> becomes <code>\"a/b\"</code>.<li>Output-typed"
                    + " attributes become the part of the label after the package, including the"
                    + " file extension (for the above example, <code>\"a/b.c\"</code>).<li>All"
                    + " list-typed attributes (for example, <code>attr.label_list</code>) used in"
                    + " placeholders are required to have <i>exactly one element</i>. Their"
                    + " conversion is the same as their non-list version"
                    + " (<code>attr.label</code>).<li>Other attribute types may not appear in"
                    + " placeholders.<li>The special non-attribute placeholders"
                    + " <code>%{dirname}</code> and <code>%{basename}</code> expand to those parts"
                    + " of the rule's label, excluding its package. For example, in"
                    + " <code>\"//pkg:a/b.c\"</code>, the dirname is <code>a</code> and the"
                    + " basename is <code>b.c</code>.</ul><p>In practice, the most common"
                    + " substitution placeholder is <code>\"%{name}\"</code>. For example, for a"
                    + " target named \"foo\", the outputs dict <code>{\"bin\":"
                    + " \"%{name}.exe\"}</code> predeclares an output named <code>foo.exe</code>"
                    + " that is accessible in the implementation function as"
                    + " <code>ctx.outputs.bin</code>."),
        @Param(
            name = "executable",
            named = true,
            defaultValue = "False",
            doc =
                "Whether this rule is considered executable, that is, whether it may be the "
                    + "subject of a <code>blaze run</code> command. See the "
                    + "<a href='../rules.$DOC_EXT#executable-rules-and-test-rules'>Rules page</a> "
                    + "for more information."),
        @Param(
            name = "output_to_genfiles",
            named = true,
            defaultValue = "False",
            doc =
                "If true, the files will be generated in the genfiles directory instead of the "
                    + "bin directory. Unless you need it for compatibility with existing rules "
                    + "(e.g. when generating header files for C++), do not set this flag."),
        @Param(
            name = "fragments",
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            named = true,
            defaultValue = "[]",
            doc =
                "List of names of configuration fragments that the rule requires "
                    + "in target configuration."),
        @Param(
            name = "host_fragments",
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            named = true,
            defaultValue = "[]",
            doc =
                "List of names of configuration fragments that the rule requires "
                    + "in host configuration."),
        @Param(
            name = "_skylark_testable",
            named = true,
            defaultValue = "False",
            doc =
                "<i>(Experimental)</i><br/><br/>"
                    + "If true, this rule will expose its actions for inspection by rules that "
                    + "depend on it via an <a href=\"globals.html#Actions\">Actions</a> "
                    + "provider. The provider is also available to the rule itself by calling "
                    + "<a href=\"ctx.html#created_actions\">ctx.created_actions()</a>."
                    + "<br/><br/>"
                    + "This should only be used for testing the analysis-time behavior of "
                    + "Starlark rules. This flag may be removed in the future."),
        @Param(
            name = TOOLCHAINS_PARAM,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            named = true,
            defaultValue = "[]",
            doc =
                "If set, the set of toolchains this rule requires. Toolchains will be "
                    + "found by checking the current platform, and provided to the rule "
                    + "implementation via <code>ctx.toolchain</code>."),
        @Param(
            name = "incompatible_use_toolchain_transition",
            defaultValue = "False",
            named = true,
            doc =
                "If set, this rule will use the toolchain transition for toolchain dependencies."
                    + " This is ignored if the --incompatible_use_toolchain_transition flag is"
                    + " set."),
        @Param(
            name = "doc",
            named = true,
            defaultValue = "''",
            doc =
                "A description of the rule that can be extracted by documentation generating "
                    + "tools."),
        @Param(
            name = "provides",
            named = true,
            positional = false,
            defaultValue = "[]",
            doc = PROVIDES_DOC),
        @Param(
            name = EXEC_COMPATIBLE_WITH_PARAM,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            named = true,
            positional = false,
            defaultValue = "[]",
            doc =
                "A list of constraints on the execution platform that apply to all targets of "
                    + "this rule type."),
        @Param(
            name = "analysis_test",
            named = true,
            positional = false,
            defaultValue = "False",
            doc =
                "If true, then this rule is treated as an analysis test. <p>Note: Analysis test"
                    + " rules are primarily defined using infrastructure provided in core Starlark"
                    + " libraries. See <a href=\"../testing.html#for-testing-rules\">Testing</a>"
                    + " for guidance. <p>If a rule is defined as an analysis test rule, it becomes"
                    + " allowed to use configuration transitions defined using <a"
                    + " href=\"#analysis_test_transition\">analysis_test_transition</a> on its"
                    + " attributes, but opts into some restrictions: <ul><li>Targets of this rule"
                    + " are limited in the number of transitive dependencies they may have."
                    + " <li>The rule is considered a test rule (as if <code>test=True</code> were"
                    + " set). This supercedes the value of <code>test</code></li> <li>The rule"
                    + " implementation function may not register actions."
                    + " Instead, it must register a pass/fail result via providing <a"
                    + " href='AnalysisTestResultInfo.html'>AnalysisTestResultInfo</a>.</li></ul>"),
        @Param(
            name = "build_setting",
            allowedTypes = {
              @ParamType(type = BuildSettingApi.class),
              @ParamType(type = NoneType.class),
            },
            defaultValue = "None",
            named = true,
            positional = false,
            doc =
                "If set, describes what kind of "
                    + "<a href = '../config.$DOC_EXT#user-defined-build-settings'><code>build "
                    + "setting</code></a> this rule is. See the "
                    + "<a href='config.html'><code>config</code></a> module. If this is "
                    + "set, a mandatory attribute named \"build_setting_default\" is automatically "
                    + "added to this rule, with a type corresponding to the value passed in here."),
        @Param(
            name = "cfg",
            defaultValue = "None",
            named = true,
            positional = false,
            doc =
                "If set, points to the configuration transition the rule will "
                    + "apply to its own configuration before analysis."),
        @Param(
            name = "exec_groups",
            allowedTypes = {
              @ParamType(type = Dict.class),
              @ParamType(type = NoneType.class),
            },
            named = true,
            defaultValue = "None",
            positional = false,
            doc =
                "Dict of execution group name (string) to <a"
                    + " href='globals.html#exec_group'><code>exec_group</code>s</a>. If set,"
                    + " allows rules to run actions on multiple execution platforms within a"
                    + " single target. See <a href='../../exec-groups.html'>execution groups"
                    + " documentation</a> for more info."),
        @Param(
            name = "compile_one_filetype",
            defaultValue = "None",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = NoneType.class),
            },
            named = true,
            positional = false,
            doc =
                "Used by --compile_one_dependency: if multiple rules consume the specified file, "
                    + "should we choose this rule over others."),
      },
      useStarlarkThread = true)
  StarlarkCallable rule(
      StarlarkFunction implementation,
      Boolean test,
      Object attrs,
      Object implicitOutputs,
      Boolean executable,
      Boolean outputToGenfiles,
      Sequence<?> fragments,
      Sequence<?> hostFragments,
      Boolean starlarkTestable,
      Sequence<?> toolchains,
      boolean useToolchainTransition,
      String doc,
      Sequence<?> providesArg,
      Sequence<?> execCompatibleWith,
      Object analysisTest,
      Object buildSetting,
      Object cfg,
      Object execGroups,
      Object compileOneFiletype,
      StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(
      name = "aspect",
      doc =
          "Creates a new aspect. The result of this function must be stored in a global value. "
              + "Please see the <a href=\"../aspects.md\">introduction to Aspects</a> for more "
              + "details.",
      parameters = {
        @Param(
            name = "implementation",
            named = true,
            doc =
                "A Starlark function that implements this aspect, with exactly two parameters: "
                    + "<a href=\"Target.html\">Target</a> (the target to which the aspect is "
                    + "applied) and <a href=\"ctx.html\">ctx</a> (the rule context which the target"
                    + "is created from). Attributes of the target are available via the "
                    + "<code>ctx.rule</code> field. This function is evaluated during the "
                    + "analysis phase for each application of an aspect to a target."),
        @Param(
            name = "attr_aspects",
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            named = true,
            defaultValue = "[]",
            doc =
                "List of attribute names. The aspect propagates along dependencies specified in "
                    + " the attributes of a target with these names. Common values here include "
                    + "<code>deps</code> and <code>exports</code>. The list can also contain a "
                    + "single string <code>\"*\"</code> to propagate along all dependencies of a "
                    + "target."),
        @Param(
            name = "attrs",
            allowedTypes = {
              @ParamType(type = Dict.class),
              @ParamType(type = NoneType.class),
            },
            named = true,
            defaultValue = "None",
            doc =
                "A dictionary declaring all the attributes of the aspect. It maps from an "
                    + "attribute name to an attribute object, like `attr.label` or `attr.string` "
                    + "(see <a href=\"attr.html\">attr</a> module). Aspect attributes are "
                    + "available to implementation function as fields of <code>ctx</code> "
                    + "parameter. "
                    + ""
                    + "<p>Implicit attributes starting with <code>_</code> must have default "
                    + "values, and have type <code>label</code> or <code>label_list</code>. "
                    + ""
                    + "<p>Explicit attributes must have type <code>string</code>, and must use "
                    + "the <code>values</code> restriction. Explicit attributes restrict the "
                    + "aspect to only be used with rules that have attributes of the same "
                    + "name, type, and valid values according to the restriction."),
        @Param(
            name = "required_providers",
            named = true,
            defaultValue = "[]",
            doc =
                "This attribute allows the aspect to limit its propagation to only the targets "
                    + "whose rules advertise its required providers. The value must be a "
                    + "list containing either individual providers or lists of providers but not "
                    + "both. For example, <code>[[FooInfo], [BarInfo], [BazInfo, QuxInfo]]</code> "
                    + "is a valid value while <code>[FooInfo, BarInfo, [BazInfo, QuxInfo]]</code> "
                    + "is not valid."
                    + ""
                    + "<p>An unnested list of providers will automatically be converted to a list "
                    + "containing one list of providers. That is, <code>[FooInfo, BarInfo]</code> "
                    + "will automatically be converted to <code>[[FooInfo, BarInfo]]</code>."
                    + ""
                    + "<p>To make some rule (e.g. <code>some_rule</code>) targets visible to an "
                    + "aspect, <code>some_rule</code> must advertise all providers from at least "
                    + "one of the required providers lists. For example, if the "
                    + "<code>required_providers</code> of an aspect are "
                    + "<code>[[FooInfo], [BarInfo], [BazInfo, QuxInfo]]</code>, this aspect can "
                    + "only see <code>some_rule</code> targets if and only if "
                    + "<code>some_rule</code> provides <code>FooInfo</code> *or* "
                    + "<code>BarInfo</code> *or* both <code>BazInfo</code> *and* "
                    + "<code>QuxInfo</code>."),
        @Param(
            name = "required_aspect_providers",
            named = true,
            defaultValue = "[]",
            doc =
                "This attribute allows this aspect to inspect other aspects. The value must be a "
                    + "list containing either individual providers or lists of providers but not "
                    + "both. For example, <code>[[FooInfo], [BarInfo], [BazInfo, QuxInfo]]</code> "
                    + "is a valid value while <code>[FooInfo, BarInfo, [BazInfo, QuxInfo]]</code> "
                    + "is not valid."
                    + ""
                    + "<p>An unnested list of providers will automatically be converted to a list "
                    + "containing one list of providers. That is, "
                    + "<code>[FooInfo, BarInfo]</code> will automatically be converted to "
                    + "<code>[[FooInfo, BarInfo]]</code>. "
                    + ""
                    + "<p>To make another aspect (e.g. <code>other_aspect</code>) visible to this "
                    + "aspect, <code>other_aspect</code> must provide all providers from at least "
                    + "one of the lists. In the example of "
                    + "<code>[[FooInfo], [BarInfo], [BazInfo, QuxInfo]]</code>, this aspect can "
                    + "only see <code>other_aspect</code> if and only if <code>other_aspect</code> "
                    + "provides <code>FooInfo</code> *or* <code>BarInfo</code> *or* both "
                    + "<code>BazInfo</code> *and* <code>QuxInfo</code>."),
        @Param(name = "provides", named = true, defaultValue = "[]", doc = PROVIDES_DOC),
        @Param(
            name = "requires",
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = StarlarkAspectApi.class)},
            named = true,
            enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_REQUIRED_ASPECTS,
            defaultValue = "[]",
            valueWhenDisabled = "[]",
            doc = "(Experimental) List of aspects required to be propagated before this aspect."),
        @Param(
            name = "fragments",
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            named = true,
            defaultValue = "[]",
            doc =
                "List of names of configuration fragments that the aspect requires "
                    + "in target configuration."),
        @Param(
            name = "host_fragments",
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            named = true,
            defaultValue = "[]",
            doc =
                "List of names of configuration fragments that the aspect requires "
                    + "in host configuration."),
        @Param(
            name = TOOLCHAINS_PARAM,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            named = true,
            defaultValue = "[]",
            doc =
                "If set, the set of toolchains this rule requires. Toolchains will be "
                    + "found by checking the current platform, and provided to the rule "
                    + "implementation via <code>ctx.toolchain</code>."),
        @Param(
            name = "incompatible_use_toolchain_transition",
            defaultValue = "False",
            named = true,
            doc =
                "If set, this aspect will use the toolchain transition for toolchain dependencies."
                    + " This is ignored if the --incompatible_use_toolchain_transition flag is"
                    + " set."),
        @Param(
            name = "doc",
            named = true,
            defaultValue = "''",
            doc =
                "A description of the aspect that can be extracted by documentation generating "
                    + "tools."),
        @Param(
            name = "apply_to_generating_rules",
            named = true,
            positional = false,
            defaultValue = "False",
            doc =
                "If true, the aspect will, when applied to an output file, instead apply to the "
                    + "output file's generating rule. "
                    + "<p>For example, suppose an aspect propagates transitively through attribute "
                    + "`deps` and it is applied to target `alpha`. Suppose `alpha` has "
                    + "`deps = [':beta_output']`, where `beta_output` is a declared output of "
                    + "a target `beta`. Suppose `beta` has a target `charlie` as one of its "
                    + "`deps`. If `apply_to_generating_rules=True` for the aspect, then the aspect "
                    + "will propagate through `alpha`, `beta`, and `charlie`. If False, then the "
                    + "aspect will propagate only to `alpha`. </p><p>False by default.</p>")
      },
      useStarlarkThread = true)
  StarlarkAspectApi aspect(
      StarlarkFunction implementation,
      Sequence<?> attributeAspects,
      Object attrs,
      Sequence<?> requiredProvidersArg,
      Sequence<?> requiredAspectProvidersArg,
      Sequence<?> providesArg,
      Sequence<?> requiredAspects,
      Sequence<?> fragments,
      Sequence<?> hostFragments,
      Sequence<?> toolchains,
      boolean useToolchainTransition,
      String doc,
      Boolean applyToGeneratingRules,
      StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(
      name = "Label",
      doc =
          "Creates a Label referring to a BUILD target. Use "
              + "this function only when you want to give a default value for the label "
              + "attributes. The argument must refer to an absolute label. "
              + "Example: <br><pre class=language-python>Label(\"//tools:default\")</pre>",
      parameters = {
        @Param(name = "label_string", doc = "the label string."),
        @Param(
            name = "relative_to_caller_repository",
            defaultValue = "False",
            named = true,
            positional = false,
            doc =
                "Deprecated. Do not use. "
                    + "When relative_to_caller_repository is True and the calling thread is a "
                    + "rule's implementation function, then a repo-relative label //foo:bar is "
                    + "resolved relative to the rule's repository.  For calls to Label from any "
                    + "other thread, or calls in which the relative_to_caller_repository flag is "
                    + "False, a repo-relative label is resolved relative to the file in which the "
                    + "Label() call appears.")
      },
      useStarlarkThread = true)
  @StarlarkConstructor
  Label label(String labelString, Boolean relativeToCallerRepository, StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(
      name = "exec_group",
      doc =
          "<i>experimental</i> Creates an <a href='../../exec-groups.html'>execution group</a> "
              + "which can be used to create actions for a specific execution platform during rule "
              + "implementation.",
      parameters = {
        @Param(
            name = TOOLCHAINS_PARAM,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            named = true,
            positional = false,
            defaultValue = "[]",
            doc = "<i>Experimental</i> The set of toolchains this execution group requires."),
        @Param(
            name = EXEC_COMPATIBLE_WITH_PARAM,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            named = true,
            positional = false,
            defaultValue = "[]",
            doc = "<i>Experimental</i> A list of constraints on the execution platform."),
        @Param(
            name = "copy_from_rule",
            defaultValue = "False",
            named = true,
            positional = false,
            doc =
                "<i>Experimental</i> If set to true, this exec group inherits the toolchains and "
                    + "constraints of the rule to which this group is attached. If set to any "
                    + "other string this will throw an error.")
      },
      useStarlarkThread = true)
  ExecGroupApi execGroup(
      Sequence<?> execCompatibleWith,
      Sequence<?> toolchains,
      Boolean copyFromRule,
      StarlarkThread thread)
      throws EvalException;
}
