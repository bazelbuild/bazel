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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.NoneType;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.SkylarkIndexable;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import com.google.devtools.build.lib.syntax.Tuple;
import javax.annotation.Nullable;

/** Interface for a context object given to rule implementation functions. */
@SkylarkModule(
    name = "ctx",
    category = SkylarkModuleCategory.BUILTIN,
    doc =
        "A context object that is passed to the implementation function for a rule or aspect. It"
            + " provides access to the information and methods needed to analyze the current"
            + " target.<p>In particular, it lets the implementation function access the current"
            + " target's label, attributes, configuration, and the providers of its dependencies."
            + " It has methods for declaring output files and the actions that produce them."
            + "<p>Context objects essentially live for the duration of the call to the"
            + " implementation function. It is not useful to access these objects outside of their"
            + " associated function. See the <a"
            + " href='../rules.$DOC_EXT#implementation-function'>Rules page</a> for more "
            + "information.")
public interface SkylarkRuleContextApi extends StarlarkValue {

  public static final String DOC_NEW_FILE_TAIL = "Does not actually create a file on the file "
      + "system, just declares that some action will do so. You must create an action that "
      + "generates the file. If the file should be visible to other rules, declare a rule output "
      + "instead when possible. Doing so enables Blaze to associate a label with the file that "
      + "rules can refer to (allowing finer dependency control) instead of referencing the whole "
      + "rule.";
  public static final String EXECUTABLE_DOC =
      "A <code>struct</code> containing executable files defined in <a "
          + "href='attr.html#label'>label type attributes</a> marked as <a "
          + "href='attr.html#label.executable'><code>executable=True</code><a>. The struct "
          + "fields correspond to the attribute names. Each value in the struct is either a <a "
          + "href='File.html'><code>File</code></a> or <code>None</code>. If an optional "
          + "attribute is not specified in the rule then the corresponding struct value is "
          + "<code>None</code>. If a label type is not marked as <code>executable=True</code>, no "
          + "corresponding struct field is generated. <a "
          + "href=\"https://github.com/bazelbuild/examples/blob/master/rules/actions_run/"
          + "execute.bzl\">See example of use</a>.";
  public static final String FILES_DOC =
      "A <code>struct</code> containing files defined in <a href='attr.html#label'>label</a>"
          + " or <a href='attr.html#label_list'>label list</a> type attributes. The struct"
          + " fields correspond to the attribute names. The struct values are <code>list</code> of"
          + " <a href='File.html'><code>File</code></a>s.  It is a shortcut for:<pre"
          + " class=language-python>[f for t in ctx.attr.&lt;ATTR&gt; for f in t.files]</pre> In"
          + " other words, use <code>files</code> to access the <a"
          + " href=\"../rules.$DOC_EXT#requesting-output-files\">default outputs</a> of a"
          + " dependency. <a"
          + " href=\"https://github.com/bazelbuild/examples/blob/master/rules/depsets/foo.bzl\">See"
          + " example of use</a>.";
  public static final String FILE_DOC =
      "A <code>struct</code> containing files defined in <a href='attr.html#label'>label type"
          + " attributes</a> marked as <a"
          + " href='attr.html#label.allow_single_file'><code>allow_single_file</code></a>. The"
          + " struct fields correspond to the attribute names. The struct value is always a <a"
          + " href='File.html'><code>File</code></a> or <code>None</code>. If an optional"
          + " attribute is not specified in the rule then the corresponding struct value is"
          + " <code>None</code>. If a label type is not marked as <code>allow_single_file</code>,"
          + " no corresponding struct field is generated. It is a shortcut for:<pre"
          + " class=language-python>list(ctx.attr.&lt;ATTR&gt;.files)[0]</pre>In other words, use"
          + " <code>file</code> to access the (singular) <a"
          + " href=\"../rules.$DOC_EXT#requesting-output-files\">default output</a> of a"
          + " dependency. <a"
          + " href=\"https://github.com/bazelbuild/examples/blob/master/rules/expand_template/hello.bzl\">See"
          + " example of use</a>.";
  public static final String ATTR_DOC =
      "A struct to access the values of the <a href='../rules.$DOC_EXT#attributes'>attributes</a>. "
          + "The values are provided by the user (if not, a default value is used). The attributes "
          + "of the struct and the types of their values correspond to the keys and values of the "
          + "<a href='globals.html#rule.attrs'><code>attrs</code> dict</a> provided to the <a "
          + "href='globals.html#rule'><code>rule</code> function</a>. <a "
          + "href=\"https://github.com/bazelbuild/examples/blob/master/rules/attributes/"
          + "printer.bzl\">See example of use</a>.";
  public static final String SPLIT_ATTR_DOC =
      "A struct to access the values of attributes with split configurations. If the attribute is "
          + "a label list, the value of split_attr is a dict of the keys of the split (as strings) "
          + "to lists of the ConfiguredTargets in that branch of the split. If the attribute is a "
          + "label, then the value of split_attr is a dict of the keys of the split (as strings) "
          + "to single ConfiguredTargets. Attributes with split configurations still appear in the "
          + "attr struct, but their values will be single lists with all the branches of the split "
          + "merged together.";
  public static final String OUTPUTS_DOC =
      "A pseudo-struct containing all the predeclared output files, represented by "
          + "<a href='File.html'><code>File</code></a> objects. See the "
          + "<a href='../rules.$DOC_EXT#files'>Rules page</a> for more information and examples."
          + "<p>This field does not exist on aspect contexts, since aspects do not have "
          + "predeclared outputs."
          + "<p>The fields of this object are defined as follows. It is an error if two outputs "
          + "produce the same field name or have the same label."
          + "<ul>"
          + "<li>If the rule declares an <a href='globals.html#rule.outputs'><code>outputs</code>"
          + "</a> dict, then for every entry in the dict, there is a field whose name is the key "
          + "and whose value is the corresponding <code>File</code>."
          + "<li>For every attribute of type <a href='attr.html#output'><code>attr.output</code>"
          + "</a> that the rule declares, there is a field whose name is the attribute's name. "
          + "If the target specified a label for that attribute, then the field value is the "
          + "corresponding <code>File</code>; otherwise the field value is <code>None</code>."
          + "<li>For every attribute of type <a href='attr.html#output_list'><code>attr.output_list"
          + "</code></a> that the rule declares, there is a field whose name is the attribute's "
          + "name. The field value is a list of <code>File</code> objects corresponding to the "
          + "labels given for that attribute in the target, or an empty list if the attribute was "
          + "not specified in the target."
          + "<li><b>(Deprecated)</b> If the rule is marked <a href='globals.html#rule.executable'>"
          + "<code>executable</code></a> or <a href='globals.html#rule.test'><code>test</code>"
          + "</a>, there is a field named <code>\"executable\"</code>, which is the default "
          + "executable. It is recommended that instead of using this, you pass another file "
          + "(either predeclared or not) to the <code>executable</code> arg of "
          + "<a href='DefaultInfo.html'><code>DefaultInfo</code></a>."
          + "</ul>";

  @SkylarkCallable(
      name = "default_provider",
      structField = true,
      doc = "Deprecated. Use <a href=\"DefaultInfo.html\">DefaultInfo</a> instead.")
  public ProviderApi getDefaultProvider();

  @SkylarkCallable(
    name = "actions",
    structField = true,
    doc = "Contains methods for declaring output files and the actions that produce them."
  )
  public SkylarkActionFactoryApi actions();

  @SkylarkCallable(
      name = "created_actions",
      doc =
          "For rules with <a href=\"globals.html#rule._skylark_testable\">_skylark_testable</a>"
              + " set to <code>True</code>, this returns an <a"
              + " href=\"globals.html#Actions\">Actions</a> provider representing all actions"
              + " created so far for the current rule. For all other rules, returns"
              + " <code>None</code>. Note that the provider is not updated when subsequent actions"
              + " are created, so you will have to call this function again if you wish to inspect"
              + " them. <br/><br/>This is intended to help write tests for rule-implementation"
              + " helper functions, which may take in a <code>ctx</code> object and create actions"
              + " on it.")
  public StarlarkValue createdActions() throws EvalException;

  @SkylarkCallable(name = "attr", structField = true, doc = ATTR_DOC)
  public StructApi getAttr() throws EvalException;

  @SkylarkCallable(name = "split_attr", structField = true, doc = SPLIT_ATTR_DOC)
  public StructApi getSplitAttr() throws EvalException;

  @SkylarkCallable(name = "executable", structField = true, doc = EXECUTABLE_DOC)
  public StructApi getExecutable() throws EvalException;

  @SkylarkCallable(name = "file", structField = true, doc = FILE_DOC)
  public StructApi getFile() throws EvalException;

  @SkylarkCallable(name = "files", structField = true, doc = FILES_DOC)
  public StructApi getFiles() throws EvalException;

  @SkylarkCallable(
    name = "workspace_name",
    structField = true,
    doc = "Returns the workspace name as defined in the WORKSPACE file."
  )
  public String getWorkspaceName() throws EvalException;

  @SkylarkCallable(
    name = "label",
    structField = true,
    doc = "The label of the target currently being analyzed."
  )
  public Label getLabel() throws EvalException;

  @SkylarkCallable(
    name = "fragments",
    structField = true,
    doc = "Allows access to configuration fragments in target configuration."
  )
  public FragmentCollectionApi getFragments() throws EvalException;

  @SkylarkCallable(
    name = "host_fragments",
    structField = true,
    doc = "Allows access to configuration fragments in host configuration."
  )
  public FragmentCollectionApi getHostFragments() throws EvalException;

  @SkylarkCallable(
    name = "configuration",
    structField = true,
    doc =
        "Returns the default configuration. See the <a href=\"configuration.html\">"
            + "configuration</a> type for more details."
  )
  public BuildConfigurationApi getConfiguration() throws EvalException;

  @SkylarkCallable(
    name = "host_configuration",
    structField = true,
    doc =
        "Returns the host configuration. See the <a href=\"configuration.html\">"
            + "configuration</a> type for more details."
  )
  public BuildConfigurationApi getHostConfiguration() throws EvalException;

  @SkylarkCallable(
      name = "build_setting_value",
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_BUILD_SETTING_API,
      doc =
          "<b>Experimental. This field is experimental and subject to change at any time. Do not "
              + "depend on it.</b> <p>Returns the value of the build setting that is represented "
              + "by the current target. It is an error to access this field for rules that do not "
              + "set the <code>build_setting</code> attribute in their rule definition.")
  public Object getBuildSettingValue() throws EvalException;

  @SkylarkCallable(
      name = "coverage_instrumented",
      doc =
          "Returns whether code coverage instrumentation should be generated when performing "
              + "compilation actions for this rule or, if <code>target</code> is provided, the "
              + "rule specified by that Target. (If a non-rule or a Starlark rule Target is "
              + "provided, this returns False.) Checks if the sources of the current rule "
              + "(if no Target is provided) or the sources of Target should be instrumented "
              + "based on the --instrumentation_filter and "
              + "--instrument_test_targets config settings. "
              + "This differs from <code>coverage_enabled</code> in the "
              + "<a href=\"configuration.html\">configuration</a>, which notes whether coverage "
              + "data collection is enabled for the entire run, but not whether a specific "
              + "target should be instrumented.",
      parameters = {
        @Param(
            name = "target",
            type = TransitiveInfoCollectionApi.class,
            defaultValue = "None",
            noneable = true,
            named = true,
            doc = "A Target specifying a rule. If not provided, defaults to the current rule.")
      })
  public boolean instrumentCoverage(Object targetUnchecked) throws EvalException;

  @SkylarkCallable(
    name = "features",
    structField = true,
    doc = "Returns the set of features that are explicitly enabled by the user for this rule."
  )
  public ImmutableList<String> getFeatures() throws EvalException;

  @SkylarkCallable(
      name = "disabled_features",
      structField = true,
      doc = "Returns the set of features that are explicitly disabled by the user for this rule.")
  ImmutableList<String> getDisabledFeatures() throws EvalException;

  @SkylarkCallable(
    name = "bin_dir",
    structField = true,
    doc = "The root corresponding to bin directory."
  )
  public FileRootApi getBinDirectory() throws EvalException;

  @SkylarkCallable(
    name = "genfiles_dir",
    structField = true,
    doc = "The root corresponding to genfiles directory."
  )
  public FileRootApi getGenfilesDirectory() throws EvalException;

  @SkylarkCallable(
    name = "outputs",
    structField = true,
    doc = OUTPUTS_DOC
  )
  public ClassObject outputs() throws EvalException;

  @SkylarkCallable(
    name = "rule",
    structField = true,
    doc =
        "Returns rule attributes descriptor for the rule that aspect is applied to."
            + " Only available in aspect implementation functions."
  )
  public SkylarkAttributesCollectionApi rule() throws EvalException;

  @SkylarkCallable(
    name = "aspect_ids",
    structField = true,
    doc =
        "Returns a list ids for all aspects applied to the target."
            + " Only available in aspect implementation functions."
  )
  public ImmutableList<String> aspectIds() throws EvalException;

  @SkylarkCallable(
      name = "var",
      structField = true,
      doc = "Dictionary (String to String) of configuration variables.")
  public Dict<String, String> var() throws EvalException;

  @SkylarkCallable(
    name = "toolchains",
    structField = true,
    doc = "Toolchains required for this rule."
  )
  public SkylarkIndexable toolchains() throws EvalException;

  @SkylarkCallable(
      name = "tokenize",
      doc = "Splits a shell command into a list of tokens.",
      // TODO(cparsons): Look into flipping this to true.
      documented = false,
      parameters = {
        @Param(
            name = "option",
            positional = true,
            named = false,
            type = String.class,
            doc = "The string to split."),
      })
  public Sequence<String> tokenize(String optionString) throws EvalException;

  @SkylarkCallable(
      name = "expand",
      doc =
          "Expands all references to labels embedded within a string for all files using a mapping "
              + "from definition labels (i.e. the label in the output type attribute) to files. "
              + "Deprecated.",
      // TODO(cparsons): Look into flipping this to true.
      documented = false,
      parameters = {
        @Param(
            name = "expression",
            positional = true,
            named = false,
            type = String.class,
            doc = "The string expression to expand."),
        @Param(
            name = "files",
            positional = true,
            named = false,
            type = Sequence.class,
            doc = "The list of files."),
        @Param(
            name = "label_resolver",
            positional = true,
            named = false,
            type = Label.class,
            doc = "The label resolver."),
      })
  public String expand(
      @Nullable String expression,
      Sequence<?> artifacts, // <FileT>
      Label labelResolver)
      throws EvalException;

  @SkylarkCallable(
      name = "new_file",
      doc =
          "DEPRECATED. Use <a href=\"actions.html#declare_file\">ctx.actions.declare_file</a>. <br>"
              + "Creates a file object. There are four possible signatures to this method:<br><ul>"
              + ""
              + "<li>new_file(filename): Creates a file object with the given filename in the "
              + "current package.</li>"
              + ""
              + "<li>new_file(file_root, filename): Creates a file object with the given "
              + "filename under the given file root.</li>"
              + ""
              + "<li>new_file(sibling_file, filename): Creates a file object in the same "
              + "directory as the given sibling file.</li>"
              + ""
              + "<li>new_file(file_root, sibling_file, suffix): Creates a file object with same "
              + "base name of the sibling_file but with different given suffix, under the given "
              + "file root.</li></ul> <br>"
              + DOC_NEW_FILE_TAIL,
      parameters = {
          @Param(
              name = "var1",
              allowedTypes = {
                  @ParamType(type = String.class),
                  @ParamType(type = FileRootApi.class),
                  @ParamType(type = FileApi.class),
              },
              doc = ""
          ),
          @Param(
              name = "var2",
              allowedTypes = {
                  @ParamType(type = String.class),
                  @ParamType(type = FileApi.class),
              },
              defaultValue = "unbound",
              doc = ""
          ),
          @Param(
              name = "var3",
              allowedTypes = {
                  @ParamType(type = String.class),
              },
              defaultValue = "unbound",
              doc = ""
          )
      },
      useLocation = true
  )
  public FileApi newFile(Object var1,
      Object var2,
      Object var3,
      Location loc) throws EvalException;

  @SkylarkCallable(
    name = "experimental_new_directory",
    documented = false,
    parameters = {
      @Param(name = "name", type = String.class),
      @Param(
        name = "sibling",
        type = FileApi.class,
        defaultValue = "None",
        noneable = true,
        named = true
      )
    }
  )
  public FileApi newDirectory(String name, Object siblingArtifactUnchecked) throws EvalException;

  @SkylarkCallable(
      name = "check_placeholders",
      documented = false,
      parameters = {
        @Param(
            name = "template",
            positional = true,
            named = false,
            type = String.class,
            doc = "The template."),
        @Param(
            name = "allowed_placeholders",
            positional = true,
            named = false,
            type = Sequence.class,
            doc = "The allowed placeholders."),
      })
  public boolean checkPlaceholders(String template, Sequence<?> allowedPlaceholders) // <String>
      throws EvalException;

  @SkylarkCallable(
      name = "expand_make_variables",
      doc =
          "<b>Deprecated.</b> Use <a href=\"ctx.html#var\">ctx.var</a> to access the variables "
              + "instead.<br>Returns a string after expanding all references to \"Make "
              + "variables\". The "
              + "variables must have the following format: <code>$(VAR_NAME)</code>. Also, "
              + "<code>$$VAR_NAME</code> expands to <code>$VAR_NAME</code>. "
              + "Examples:"
              + "<pre class=language-python>\n"
              + "ctx.expand_make_variables(\"cmd\", \"$(MY_VAR)\", {\"MY_VAR\": \"Hi\"})  "
              + "# == \"Hi\"\n"
              + "ctx.expand_make_variables(\"cmd\", \"$$PWD\", {})  # == \"$PWD\"\n"
              + "</pre>"
              + "Additional variables may come from other places, such as configurations. Note "
              + "that this function is experimental.",
      parameters = {
        @Param(
            name = "attribute_name",
            positional = true,
            named = false,
            type = String.class,
            doc = "The attribute name. Used for error reporting."),
        @Param(
            name = "command",
            positional = true,
            named = false,
            type = String.class,
            doc =
                "The expression to expand. It can contain references to " + "\"Make variables\"."),
        @Param(
            name = "additional_substitutions",
            positional = true,
            named = false,
            type = Dict.class,
            doc = "Additional substitutions to make beyond the default make variables."),
      })
  public String expandMakeVariables(
      String attributeName,
      String command,
      final Dict<?, ?> additionalSubstitutions) // <String, String>
      throws EvalException;

  @SkylarkCallable(
    name = "info_file",
    structField = true,
    documented = false,
    doc =
        "Returns the file that is used to hold the non-volatile workspace status for the "
            + "current build request."
  )
  public FileApi getStableWorkspaceStatus() throws InterruptedException, EvalException;

  @SkylarkCallable(
    name = "version_file",
    structField = true,
    documented = false,
    doc =
        "Returns the file that is used to hold the volatile workspace status for the "
            + "current build request."
  )
  public FileApi getVolatileWorkspaceStatus() throws InterruptedException, EvalException;

  @SkylarkCallable(
    name = "build_file_path",
    structField = true,
    documented = true,
    doc = "Returns path to the BUILD file for this rule, relative to the source root."
  )
  public String getBuildFileRelativePath() throws EvalException;

  @SkylarkCallable(
      name = "action",
      doc =
          "DEPRECATED. Use <a href=\"actions.html#run\">ctx.actions.run()</a> or"
              + " <a href=\"actions.html#run_shell\">ctx.actions.run_shell()</a>. <br>"
              + "Creates an action that runs an executable or a shell command."
              + " You must specify either <code>command</code> or <code>executable</code>.\n"
              + "Actions and genrules are very similar, but have different use cases. Actions are "
              + "used inside rules, and genrules are used inside macros. Genrules also have make "
              + "variable expansion.",
      parameters = {
        @Param(
            name = "outputs",
            type = Sequence.class,
            generic1 = FileApi.class,
            named = true,
            positional = false,
            doc = "List of the output files of the action."),
        @Param(
            name = "inputs",
            allowedTypes = {
              @ParamType(type = Sequence.class),
              @ParamType(type = Depset.class),
            },
            generic1 = FileApi.class,
            defaultValue = "[]",
            named = true,
            positional = false,
            doc = "List of the input files of the action."),
        @Param(
            name = "executable",
            type = Object.class,
            allowedTypes = {
              @ParamType(type = FileApi.class),
              @ParamType(type = String.class),
              @ParamType(type = NoneType.class),
            },
            noneable = true,
            defaultValue = "None",
            named = true,
            positional = false,
            doc = "The executable file to be called by the action."),
        @Param(
            name = "tools",
            allowedTypes = {
              @ParamType(type = Sequence.class),
              @ParamType(type = Depset.class),
            },
            generic1 = FileApi.class,
            defaultValue = "unbound",
            named = true,
            positional = false,
            doc =
                "List of the any tools needed by the action. Tools are inputs with additional "
                    + "runfiles that are automatically made available to the action."),
        @Param(
            name = "arguments",
            allowedTypes = {
              @ParamType(type = Sequence.class),
            },
            defaultValue = "[]",
            named = true,
            positional = false,
            doc =
                "Command line arguments of the action. Must be a list of strings or actions.args() "
                    + "objects."),
        @Param(
            name = "mnemonic",
            type = String.class,
            noneable = true,
            defaultValue = "None",
            named = true,
            positional = false,
            doc = "A one-word description of the action, e.g. CppCompile or GoLink."),
        @Param(
            name = "command",
            type = Object.class,
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Sequence.class, generic1 = String.class),
              @ParamType(type = NoneType.class),
            },
            noneable = true,
            defaultValue = "None",
            named = true,
            positional = false,
            doc =
                "Shell command to execute. It is usually preferable to "
                    + "use <code>executable</code> instead. "
                    + "Arguments are available with <code>$1</code>, <code>$2</code>, etc."),
        @Param(
            name = "progress_message",
            type = String.class,
            noneable = true,
            defaultValue = "None",
            named = true,
            positional = false,
            doc =
                "Progress message to show to the user during the build, "
                    + "e.g. \"Compiling foo.cc to create foo.o\"."),
        @Param(
            name = "use_default_shell_env",
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = "Whether the action should use the built in shell environment or not."),
        @Param(
            name = "env",
            type = Dict.class,
            noneable = true,
            defaultValue = "None",
            named = true,
            positional = false,
            doc = "Sets the dictionary of environment variables."),
        @Param(
            name = "execution_requirements",
            type = Dict.class,
            noneable = true,
            defaultValue = "None",
            named = true,
            positional = false,
            doc =
                "Information for scheduling the action. See "
                    + "<a href=\"$BE_ROOT/common-definitions.html#common.tags\">tags</a> "
                    + "for useful keys."),
        @Param(
            // TODO(bazel-team): The name here isn't accurate anymore. This is technically
            // experimental,
            // so folks shouldn't be too attached, but consider renaming to be more accurate/opaque.
            name = "input_manifests",
            type = Sequence.class,
            noneable = true,
            defaultValue = "None",
            named = true,
            positional = false,
            doc =
                "(Experimental) sets the input runfiles metadata; "
                    + "they are typically generated by resolve_command.")
      },
      allowReturnNones = true,
      useLocation = true,
      useStarlarkThread = true)
  public NoneType action(
      Sequence<?> outputs,
      Object inputs,
      Object executableUnchecked,
      Object toolsUnchecked,
      Object arguments,
      Object mnemonicUnchecked,
      Object commandUnchecked,
      Object progressMessage,
      Boolean useDefaultShellEnv,
      Object envUnchecked,
      Object executionRequirementsUnchecked,
      Object inputManifestsUnchecked,
      Location loc,
      StarlarkThread thread)
      throws EvalException;

  @SkylarkCallable(
      name = "expand_location",
      doc =
          "Expands all <code>$(location ...)</code> templates in the given string by replacing "
              + "<code>$(location //x)</code> with the path of the output file of target //x. "
              + "Expansion only works for labels that point to direct dependencies of this rule or "
              + "that are explicitly listed in the optional argument <code>targets</code>. "
              + "<br/><br/>"
              //
              + "<code>$(location ...)</code> will cause an error if the referenced target has "
              + "multiple outputs. In this case, please use <code>$(locations ...)</code> since it "
              + "produces a space-separated list of output paths. It can be safely used for a "
              + "single output file, too."
              + "<br/><br/>"
              //
              + "This function is useful to let the user specify a command in a BUILD file (like"
              + " for <code>genrule</code>). In other cases, it is often better to manipulate"
              + " labels directly.",
      parameters = {
        @Param(name = "input", type = String.class, doc = "String to be expanded."),
        @Param(
            name = "targets",
            type = Sequence.class,
            generic1 = TransitiveInfoCollectionApi.class,
            defaultValue = "[]",
            named = true,
            doc = "List of targets for additional lookup information."),
      },
      allowReturnNones = true,
      useLocation = true,
      useStarlarkThread = true)
  public String expandLocation(
      String input, Sequence<?> targets, Location loc, StarlarkThread thread) throws EvalException;

  @SkylarkCallable(
      name = "file_action",
      doc =
          "DEPRECATED. Use <a href =\"actions.html#write\">ctx.actions.write</a> instead. <br>"
              + "Creates a file write action.",
      parameters = {
        @Param(name = "output", type = FileApi.class, named = true, doc = "The output file."),
        @Param(
            name = "content",
            type = String.class,
            named = true,
            doc = "The contents of the file."),
        @Param(
            name = "executable",
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            doc = "Whether the output file should be executable (default is False).")
      },
      allowReturnNones = true,
      useLocation = true,
      useStarlarkThread = true)
  public NoneType fileAction(
      FileApi output, String content, Boolean executable, Location loc, StarlarkThread thread)
      throws EvalException;

  @SkylarkCallable(
      name = "empty_action",
      doc =
          "DEPRECATED. Use <a href=\"actions.html#do_nothing\">ctx.actions.do_nothing</a> instead."
              + " <br>"
              + "Creates an empty action that neither executes a command nor produces any "
              + "output, but that is useful for inserting 'extra actions'.",
      parameters = {
        @Param(
            name = "mnemonic",
            type = String.class,
            named = true,
            positional = false,
            doc = "A one-word description of the action, e.g. CppCompile or GoLink."),
        @Param(
            name = "inputs",
            allowedTypes = {
              @ParamType(type = Sequence.class),
              @ParamType(type = Depset.class),
            },
            generic1 = FileApi.class,
            named = true,
            positional = false,
            defaultValue = "[]",
            doc = "List of the input files of the action."),
      },
      allowReturnNones = true,
      useLocation = true,
      useStarlarkThread = true)
  public NoneType emptyAction(String mnemonic, Object inputs, Location loc, StarlarkThread thread)
      throws EvalException;

  @SkylarkCallable(
      name = "template_action",
      doc =
          "DEPRECATED. "
              + "Use <a href=\"actions.html#expand_template\">ctx.actions.expand_template()</a> "
              + "instead. <br>Creates a template expansion action.",
      parameters = {
        @Param(
            name = "template",
            type = FileApi.class,
            named = true,
            positional = false,
            doc = "The template file, which is a UTF-8 encoded text file."),
        @Param(
            name = "output",
            type = FileApi.class,
            named = true,
            positional = false,
            doc = "The output file, which is a UTF-8 encoded text file."),
        @Param(
            name = "substitutions",
            type = Dict.class,
            named = true,
            positional = false,
            doc = "Substitutions to make when expanding the template."),
        @Param(
            name = "executable",
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            positional = false,
            doc = "Whether the output file should be executable (default is False).")
      },
      allowReturnNones = true,
      useLocation = true,
      useStarlarkThread = true)
  public NoneType templateAction(
      FileApi template,
      FileApi output,
      Dict<?, ?> substitutionsUnchecked,
      Boolean executable,
      Location loc,
      StarlarkThread thread)
      throws EvalException;

  @SkylarkCallable(
      name = "runfiles",
      doc = "Creates a runfiles object.",
      parameters = {
        @Param(
            name = "files",
            type = Sequence.class,
            generic1 = FileApi.class,
            named = true,
            defaultValue = "[]",
            doc = "The list of files to be added to the runfiles."),
        // TODO(bazel-team): If we have a memory efficient support for lazy list containing
        // NestedSets we can remove this and just use files = [file] + list(set)
        // Also, allow empty set for init
        @Param(
            name = "transitive_files",
            type = Depset.class,
            generic1 = FileApi.class,
            noneable = true,
            defaultValue = "None",
            named = true,
            doc =
                "The (transitive) set of files to be added to the runfiles. The depset should "
                    + "use the <code>default</code> order (which, as the name implies, is the "
                    + "default)."),
        @Param(
            name = "collect_data",
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            doc =
                "<b>Use of this parameter is not recommended. See "
                    + "<a href=\"../rules.html#runfiles\">runfiles guide</a></b>. "
                    + "<p>Whether to collect the data "
                    + "runfiles from the dependencies in srcs, data and deps attributes."),
        @Param(
            name = "collect_default",
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            doc =
                "<b>Use of this parameter is not recommended. See "
                    + "<a href=\"../rules.html#runfiles\">runfiles guide</a></b>. "
                    + "<p>Whether to collect the default "
                    + "runfiles from the dependencies in srcs, data and deps attributes."),
        @Param(
            name = "symlinks",
            type = Dict.class,
            defaultValue = "{}",
            named = true,
            doc = "The map of symlinks to be added to the runfiles, prefixed by workspace name."),
        @Param(
            name = "root_symlinks",
            type = Dict.class,
            defaultValue = "{}",
            named = true,
            doc = "The map of symlinks to be added to the runfiles.")
      },
      useLocation = true)
  public RunfilesApi runfiles(
      Sequence<?> files,
      Object transitiveFiles,
      Boolean collectData,
      Boolean collectDefault,
      Dict<?, ?> symlinks,
      Dict<?, ?> rootSymlinks,
      Location loc)
      throws EvalException;

  @SkylarkCallable(
      name = "resolve_command",
      // TODO(bazel-team): The naming here isn't entirely accurate (input_manifests is no longer
      // manifests), but this is experimental/should be opaque to the end user.
      doc =
          "<i>(Experimental)</i> Returns a tuple <code>(inputs, command, input_manifests)</code>"
              + " of the list of resolved inputs, the argv list for the resolved command, and the"
              + " runfiles metadata required to run the command, all of them suitable for passing"
              + " as the same-named arguments of the <code>ctx.action</code> method.<br/><b>Note"
              + " for Windows users</b>: this method requires Bash (MSYS2). Consider using"
              + " <code>resolve_tools()</code> instead (if that fits your needs).",
      parameters = {
        @Param(
            name = "command",
            type = String.class, // string
            defaultValue = "''",
            named = true,
            positional = false,
            doc = "Command to resolve."),
        @Param(
            name = "attribute",
            type = String.class, // string
            defaultValue = "None",
            noneable = true,
            named = true,
            positional = false,
            doc = "Name of the associated attribute for which to issue an error, or None."),
        @Param(
            name = "expand_locations",
            type = Boolean.class,
            defaultValue = "False",
            named = true,
            positional = false,
            doc =
                "Shall we expand $(location) variables? See <a"
                    + " href=\"#expand_location\">ctx.expand_location()</a> for more details."),
        @Param(
            name = "make_variables",
            type = Dict.class, // dict(string, string)
            noneable = true,
            defaultValue = "None",
            named = true,
            positional = false,
            doc = "Make variables to expand, or None."),
        @Param(
            name = "tools",
            defaultValue = "[]",
            type = Sequence.class,
            generic1 = TransitiveInfoCollectionApi.class,
            named = true,
            positional = false,
            doc = "List of tools (list of targets)."),
        @Param(
            name = "label_dict",
            type = Dict.class,
            defaultValue = "{}",
            named = true,
            positional = false,
            doc =
                "Dictionary of resolved labels and the corresponding list of Files "
                    + "(a dict of Label : list of Files)."),
        @Param(
            name = "execution_requirements",
            type = Dict.class,
            defaultValue = "{}",
            named = true,
            positional = false,
            doc =
                "Information for scheduling the action to resolve this command. See "
                    + "<a href=\"$BE_ROOT/common-definitions.html#common.tags\">tags</a> "
                    + "for useful keys."),
      },
      useLocation = true,
      useStarlarkThread = true)
  public Tuple<Object> resolveCommand(
      String command,
      Object attributeUnchecked,
      Boolean expandLocations,
      Object makeVariablesUnchecked,
      Sequence<?> tools,
      Dict<?, ?> labelDictUnchecked,
      Dict<?, ?> executionRequirementsUnchecked,
      Location loc,
      StarlarkThread thread)
      throws EvalException;

  @SkylarkCallable(
      name = "resolve_tools",
      doc =
          "Returns a tuple <code>(inputs, input_manifests)</code> of the depset of resolved inputs"
              + " and the runfiles metadata required to run the tools, both of them suitable for"
              + " passing as the same-named arguments of the <code>ctx.actions.run</code> method."
              + "<br/><br/>In contrast to <code>ctx.resolve_command</code>, this method does not"
              + " require that Bash be installed on the machine, so it's suitable for rules built"
              + " on Windows.",
      parameters = {
        @Param(
            name = "tools",
            defaultValue = "[]",
            type = Sequence.class,
            generic1 = TransitiveInfoCollectionApi.class,
            named = true,
            positional = false,
            doc = "List of tools (list of targets)."),
      },
      useLocation = false,
      useStarlarkThread = false)
  public Tuple<Object> resolveTools(Sequence<?> tools) throws EvalException;
}
