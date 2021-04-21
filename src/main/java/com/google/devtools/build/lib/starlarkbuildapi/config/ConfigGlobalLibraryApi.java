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

package com.google.devtools.build.lib.starlarkbuildapi.config;

import com.google.devtools.build.docgen.annot.DocumentMethods;
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkThread;

/** A collection of top-level Starlark functions pertaining to configuration. */
@DocumentMethods
public interface ConfigGlobalLibraryApi {
  @StarlarkMethod(
      name = "transition",
      doc =
          "A transition that reads a set of input build settings and writes a set of output build "
              + "settings."
              + "<p>Example:</p>"
              + "<p><pre class=\"language-python\">\n"
              + "def _transition_impl(settings, attr):\n"
              + "    # This transition just reads the current CPU value as a demonstration.\n"
              + "    # A real transition could incorporate this into its followup logic.\n"
              + "    current_cpu = settings[\"//command_line_option:cpu\"]\n"
              + "    return {\"//command_line_option:compilation_mode\": \"dbg\"}\n"
              + "\n"
              + "build_in_debug_mode = transition(\n"
              + "    implementation = _transition_impl,\n"
              + "    inputs = [\"//command_line_option:cpu\"],\n"
              + "    outputs = [\"//command_line_option:compilation_mode\"],\n"
              + ")"
              + "</pre></p>"
              + "<p>For more details see <a href=\"../config.html#user-defined-transitions\">"
              + "here</a>.</p>",
      parameters = {
        @Param(
            name = "implementation",
            positional = false,
            named = true,
            // TODO(cparsons): The settings dict should take actual Label objects as keys and not
            // strings. Update the documentation.
            doc =
                "The function implementing this transition. This function always has two "
                    + "parameters: <code>settings</code> and <code>attr</code>. The "
                    + "<code>settings</code> param is a dictionary whose set of keys is defined "
                    + "by the inputs parameter. So, for each build setting "
                    + "<code>--//foo=bar</code>, if <code>inputs</code> contains "
                    + "<code>//foo</code>, <code>settings</code> will "
                    + "have an entry <code>settings['//foo']='bar'</code>.<p>"
                    + "The <code>attr</code> param is a reference to <code>ctx.attr</code>. This "
                    + "gives the implementation function access to the rule's attributes to make "
                    + "attribute-parameterized transitions possible.<p>"
                    + "This function must return a <code>dict</code> from build setting identifier "
                    + "to build setting value; this represents the configuration transition: for "
                    + "each entry in the returned <code>dict</code>, the transition updates that "
                    + "setting to the new value. All other settings are unchanged. This function "
                    + "can also return a <code>list</code> of <code>dict</code>s or a "
                    + "<code>dict</code> of <code>dict</code>s in the case of a "
                    + "split transition."),
        @Param(
            name = "inputs",
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            positional = false,
            named = true,
            doc =
                "List of build settings that can be read by this transition. This becomes the "
                    + "key set of the settings parameter of the implementation function "
                    + "parameter."),
        @Param(
            name = "outputs",
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            positional = false,
            named = true,
            doc =
                "List of build settings that can be written by this transition. This must be "
                    + "a superset of the key set of the dictionary returned by this transition."),
      },
      useStarlarkThread = true)
  @StarlarkConstructor
  ConfigurationTransitionApi transition(
      StarlarkCallable implementation,
      Sequence<?> inputs, // <String> expected
      Sequence<?> outputs, // <String> expected
      StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(
      name = "analysis_test_transition",
      doc =
          "<p> Creates a configuration transition to be applied on "
              + "an analysis-test rule's dependencies. This transition may only be applied "
              + "on attributes of rules with <code>analysis_test = True</code>. Such rules are "
              + "restricted in capabilities (for example, the size of their dependency tree is "
              + "limited), so transitions created using this function are limited in potential "
              + "scope as compared to transitions created using "
              + "<a href=\"transition.html\">transition</a>. "
              + "<p>This function is primarily designed to facilitate the "
              + "<a href=\"../testing.html\">Analysis Test Framework</a> core library. See its "
              + "documentation (or its implementation) for best practices.",
      parameters = {
        @Param(
            name = "settings",
            positional = false,
            named = true,
            doc =
                "A dictionary containing information about configuration settings which "
                    + "should be set by this configuration transition. Keys are build setting "
                    + "labels and values are their new post-transition values. All other settings "
                    + "are unchanged. Use this to declare specific configuration settings that "
                    + "an analysis test requires to be set in order to pass."),
      },
      useStarlarkThread = true)
  ConfigurationTransitionApi analysisTestTransition(
      Dict<?, ?> changedSettings, // <String, String> expected
      StarlarkThread thread)
      throws EvalException;
}
