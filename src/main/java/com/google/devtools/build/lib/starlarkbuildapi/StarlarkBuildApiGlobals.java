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
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkThread;

/** A collection of global Starlark build API functions that belong in the global namespace. */
@DocumentMethods
public interface StarlarkBuildApiGlobals {

  @StarlarkMethod(
      name = "visibility",
      // TODO(b/22193153): Link to a concepts page for bzl-visibility.
      doc =
          "<i>(Experimental; enabled by <code>--experimental_bzl_visibility</code>. This feature's"
              + " API may change. Only packages that appear in"
              + " <code>--experimental_bzl_visibility_allowlist</code> are permitted to call this"
              + " function. Known issue: This feature currently may not work under bzlmod.)</i>"
              + "<p>Sets the bzl-visibility of the .bzl module currently being initialized."
              + "<p>The bzl-visibility of a .bzl module (not to be confused with target visibility)"
              + " governs whether or not a <code>load()</code> of that .bzl is permitted from"
              + " within the BUILD and .bzl files of a particular package. Allowed values include:"
              + "<ul>"
              + "<li><code>\"public\"</code> <i>(default)</i>: the .bzl can be loaded anywhere."
              + "<li><code>\"private\"</code>: the .bzl can only be loaded by files in the same"
              + " package (subpackages are excluded)."
              + "<li>a list of package paths (e.g. <code>[\"//pkg1\", \"//pkg2/subpkg\","
              + " ...]</code>): the .bzl can be loaded by files in any of the listed packages. Only"
              + " packages in the current repository may be specified; the repository \"@\" syntax"
              + " is disallowed."
              + "</ul>"
              + "<p>Generally, <code>visibility()</code> is called at the top of the .bzl file,"
              + " immediately after its <code>load()</code> statements. (It is poor style to put"
              + " this declaration later in the file or in a helper method.) It may not be called"
              + " more than once per .bzl, or after the .bzl's top-level code has finished"
              + " executing."
              + "<p>Note that a .bzl module having a public bzl-visibility does not necessarily"
              + " imply that its corresponding file target has public visibility. This means that"
              + " it's possible to be able to <code>load()</code> a .bzl file without being able to"
              + " depend on it in a <code>filegroup</code> or other target.",
      parameters = {
        @Param(
            name = "value",
            named = false,
            doc =
                "The bzl-visibility level to set. May be <code>\"public\"</code> or"
                    + " <code>\"private\"</code>.")
      },
      // Ordinarily we'd use enableOnlyWithFlag here to gate access on
      // --experimental_bzl_visibility. However, the StarlarkSemantics isn't available at the point
      // where the top-level environment is determined (see StarlarkModules#addPredeclared and
      // notice that it relies on the overload of Starlark#addMethods that uses the default
      // semantics). So instead we make this builtin unconditionally defined, but have it fail at
      // call time if used without the flag.
      useStarlarkThread = true)
  void visibility(Object value, StarlarkThread thread) throws EvalException;

  @StarlarkMethod(
      name = "configuration_field",
      // TODO(cparsons): Provide a link to documentation for available StarlarkConfigurationFields.
      doc =
          "References a late-bound default value for an attribute of type <a"
              + " href=\"attr.html#label\">label</a>. A value is 'late-bound' if it requires the"
              + " configuration to be built before determining the value. Any attribute using this"
              + " as a value must <a href=\"https://bazel.build/rules/rules#private-attributes\">be"
              + " private</a>. <p>Example usage: <p>Defining a rule attribute: <br><pre"
              + " class=language-python>'_foo':"
              + " attr.label(default=configuration_field(fragment='java', "
              + "name='toolchain'))</pre><p>Accessing in rule implementation: <br><pre"
              + " class=language-python>  def _rule_impl(ctx):\n"
              + "    foo_info = ctx.attr._foo\n"
              + "    ...</pre>",
      parameters = {
        @Param(
            name = "fragment",
            named = true,
            doc = "The name of a configuration fragment which contains the late-bound value."),
        @Param(
            name = "name",
            named = true,
            doc = "The name of the value to obtain from the configuration fragment."),
      },
      useStarlarkThread = true)
  LateBoundDefaultApi configurationField(String fragment, String name, StarlarkThread thread)
      throws EvalException;
}
