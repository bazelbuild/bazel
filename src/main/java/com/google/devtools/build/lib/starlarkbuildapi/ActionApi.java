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
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import java.io.IOException;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkValue;

/** Interface for actions in Starlark. */
@StarlarkBuiltin(
    name = "Action",
    category = DocCategory.BUILTIN,
    doc =
        "An action created during rule analysis.<p>This object is visible for the purpose of"
            + " testing, and may be obtained from an <code>Actions</code> provider. It is normally"
            + " not necessary to access <code>Action</code> objects or their fields within a rule's"
            + " implementation function. You may instead want to see the <a"
            + " href='https://bazel.build/extending/rules#actions'>Rules page</a> for a general"
            + " discussion of how to use actions when defining custom rules, or the <a"
            + " href='../builtins/actions.html'>API reference</a> for creating actions.<p>Some"
            + " fields of this object are only applicable for certain kinds of actions. Fields that"
            + " are inapplicable are set to <code>None</code>.")
public interface ActionApi extends StarlarkValue {

  @StarlarkMethod(name = "mnemonic", structField = true, doc = "The mnemonic for this action.")
  String getMnemonic();

  @StarlarkMethod(
      name = "inputs",
      doc = "A set of the input files of this action.",
      structField = true)
  Depset getStarlarkInputs();

  @StarlarkMethod(
      name = "outputs",
      doc = "A set of the output files of this action.",
      structField = true)
  Depset getStarlarkOutputs();

  @StarlarkMethod(
      name = "argv",
      doc =
          "For actions created by <a href=\"../builtins/actions.html#run\">ctx.actions.run()</a> or"
              + " <a href=\"../builtins/actions.html#run_shell\">ctx.actions.run_shell()</a>  an"
              + " immutable list of the arguments for the command line to be executed. Note that"
              + " for shell actions the first two arguments will be the shell path and"
              + " <code>\"-c\"</code>.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  Sequence<String> getStarlarkArgv() throws EvalException, InterruptedException;

  @StarlarkMethod(
      name = "args",
      doc =
          "A list of frozen <a href=\"../builtins/Args.html\">Args</a> objects containing"
              + " information about the action arguments. These objects contain accurate argument"
              + " information, including arguments involving expanded action output directories."
              + " However, <a href=\"../builtins/Args.html\">Args</a> objects are not readable in"
              + " the analysis phase. For a less accurate account of arguments which is available"
              + " in the analysis phase, see <a href=\"#argv\">argv</a>. <p>Note that some types of"
              + " actions do not yet support exposure of this field. For such action types, this is"
              + " <code>None</code>.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  Sequence<CommandLineArgsApi> getStarlarkArgs() throws EvalException, InterruptedException;

  /**
   * If the action writes a file whose content is known at analysis time, returns that content;
   * returns null otherwise.
   *
   * <p>The content might be unknown if, for instance, the action expands an {@code Args} object
   * that includes a directory ({@link TreeArtifact}) with {@code expand_directories=False}.
   *
   * @throws EvalException if there is a Starlark evaluation error, e.g. in an {@code Args} object's
   *     call to a {@code map_each} callback.
   * @throws IOException if there is a non-Starlark error in expanding an {@code Args} object.
   */
  @StarlarkMethod(
      name = "content",
      doc =
          "For actions created by <a"
              + " href=\"../builtins/actions.html#write\">ctx.actions.write()</a> or <a"
              + " href=\"../builtins/actions.html#expand_template\">ctx.actions.expand_template()</a>,"
              + " the contents of the file to be written, if those contents can be computed during "
              + " the analysis phase. The value is <code>None</code> if the contents cannot be"
              + " determined until the execution phase, such as when a directory in an <a"
              + " href=\"../builtins/Args.html\">Args</a> object needs to be expanded.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  String getStarlarkContent() throws IOException, EvalException, InterruptedException;

  @StarlarkMethod(
      name = "substitutions",
      doc =
          "For actions created by <a"
              + " href=\"../builtins/actions.html#expand_template\">ctx.actions.expand_template()</a>,"
              + " an immutable dict holding the substitution mapping.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  Dict<String, String> getStarlarkSubstitutions() throws EvalException;

  @StarlarkMethod(
      name = "env",
      structField = true,
      doc =
          "The 'fixed' environment variables for this action. This includes only environment"
              + " settings which are explicitly set by the action definition, and thus omits"
              + " settings which are only pre-set in the execution environment.")
  Dict<String, String> getEnv();

  @StarlarkMethod(
      name = "inherited_env",
      structField = true,
      doc =
          "The list of environment variable names whose values are inherited from the shell"
              + " environment for this action.")
  Sequence<String> getInheritedEnv();

  @StarlarkMethod(
      name = "execution_info",
      structField = true,
      doc =
          "The execution requirements for this action, set for this action specifically. This is a"
              + " dictionary that maps strings specifying execution info to arbitrary strings."
              + " This is in order to match the structure of execution info in other parts of the"
              + " code base; all relevant info is in the keyset. Returns None if this action does"
              + " not expose execution requirements.",
      allowReturnNones = true,
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_GOOGLE_LEGACY_API)
  @Nullable
  public Dict<String, String> getExecutionInfoDict();
}
