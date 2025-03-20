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

import com.google.devtools.build.docgen.annot.GlobalMethods;
import com.google.devtools.build.docgen.annot.GlobalMethods.Environment;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkThread;

/** A collection of global Starlark build API functions that belong in the global namespace. */
@GlobalMethods(environment = Environment.BZL)
public interface StarlarkBuildApiGlobals {

  @StarlarkMethod(
      name = "visibility",
      // TODO(b/22193153): Link to a concepts page for bzl-visibility. May require updating
      // RuleLinkExpander to correctly link to within the /concepts directory.
      doc =
          "<p>Sets the load visibility of the .bzl module currently being initialized."
              + "<p>The load visibility of a module governs whether or not other BUILD and .bzl"
              + " files may load it. (This is distinct from the target visibility of the underlying"
              + " .bzl source file, which governs whether the file may appear as a dependency of"
              + " other targets.) Load visibility works at the level of packages: To load a module"
              + " the file doing the loading must live in a package that has been granted"
              + " visibility to the module. A module can always be loaded within its own package,"
              + " regardless of its visibility."
              + "<p><code>visibility()</code> may only be called once per .bzl file, and only at"
              + " the top level, not inside a function. The preferred style is to put this call"
              + " immediately below the <code>load()</code> statements and any brief logic needed"
              + " to determine the argument."
              + "<p>If the flag <code>--check_bzl_visibility</code> is set to false, load"
              + " visibility violations will emit warnings but not fail the build.",
      parameters = {
        @Param(
            name = "value",
            named = false,
            doc =
                "A list of package specification strings, or a single package specification string."
                    + "<p>Package specifications follow the same format as for"
                    + " <code><a href='${link functions#package_group}'>package_group</a></code>,"
                    + " except that negative package specifications are not permitted. That is, a"
                    + " specification may have the forms:"
                    + "<ul>"
                    + "<li><code>\"//foo\"</code>: the package <code>//foo</code>" //
                    + "<li><code>\"//foo/...\"</code>: the package <code>//foo</code> and all of"
                    + " its subpackages." //
                    + "<li><code>\"public\"</code> or <code>\"private\"</code>: all packages or no"
                    + " packages, respectively"
                    + "</ul>"
                    + "<p>The \"@\" syntax is not allowed; all specifications are interpreted"
                    + " relative to the current module's repository."
                    + "<p>If <code>value</code> is a list of strings, the set of packages granted"
                    + " visibility to this module is the union of the packages represented by each"
                    + " specification. (An empty list has the same effect as <code>private</code>.)"
                    + " If <code>value</code> is a single string, it is treated as if it were the"
                    + " singleton list <code>[value]</code>."
                    + "<p>Note that the flags"
                    + " <code>--incompatible_package_group_has_public_syntax</code> and"
                    + " <code>--incompatible_fix_package_group_reporoot_syntax</code> have no"
                    + " effect on this argument. The <code>\"public\"</code> and <code>\"private\""
                    + "</code> values are always available, and <code>\"//...\"</code> is always"
                    + " interpreted as \"all packages in the current repository\".")
      },
      // Ordinarily we'd use enableOnlyWithFlag here to gate access on
      // --experimental_bzl_visibility. However, the StarlarkSemantics isn't available at the point
      // where the top-level environment is determined (see StarlarkGlobalsImpl#getFixedBzlToplevels
      // and notice that it relies on the overload of Starlark#addMethods that uses the default
      // semantics). So instead we make this builtin unconditionally defined, but have it fail at
      // call time if used without the flag.
      useStarlarkThread = true)
  void visibility(Object value, StarlarkThread thread) throws EvalException;

  @StarlarkMethod(
      name = "configuration_field",
      // TODO(cparsons): Provide a link to documentation for available StarlarkConfigurationFields.
      doc =
          "References a late-bound default value for an attribute of type <a"
              + " href=\"../toplevel/attr.html#label\">label</a>. A value is 'late-bound' if it"
              + " requires the configuration to be built before determining the value. Any"
              + " attribute using this as a value must <a"
              + " href=\"https://bazel.build/extending/rules#private-attributes\">be private</a>."
              + " <p>Example usage: <p>Defining a rule attribute: <br><pre"
              + " class=language-python>'_foo':"
              + " attr.label(default=configuration_field(fragment='java',"
              + " name='toolchain'))</pre><p>Accessing in rule implementation: <br><pre"
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
