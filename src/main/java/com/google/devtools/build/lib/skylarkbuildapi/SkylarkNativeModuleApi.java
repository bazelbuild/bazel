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

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.NoneType;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** Interface for a module with native rule and package helper functions. */
@SkylarkModule(
    name = "native",
    category = SkylarkModuleCategory.BUILTIN,
    doc =
        "A built-in module to support native rules and other package helper functions. "
            + "All native rules appear as functions in this module, e.g. "
            + "<code>native.cc_library</code>. "
            + "Note that the native module is only available in the loading phase "
            + "(i.e. for macros, not for rule implementations). Attributes will ignore "
            + "<code>None</code> values, and treat them as if the attribute was unset.<br>"
            + "The following functions are also available:")
public interface SkylarkNativeModuleApi extends StarlarkValue {

  @SkylarkCallable(
      name = "glob",
      doc =
          "Glob returns a list of every file in the current package that:<ul>\n"
              + "<li>Matches at least one pattern in <code>include</code>.</li>\n"
              + "<li>Does not match any of the patterns in <code>exclude</code> "
              + "(default <code>[]</code>).</li></ul>\n"
              + "If the <code>exclude_directories</code> argument is enabled "
              + "(set to <code>1</code>), files of type directory will be omitted from the "
              + "results (default <code>1</code>).",
      parameters = {
        @Param(
            name = "include",
            type = Sequence.class,
            generic1 = String.class,
            defaultValue = "[]",
            named = true,
            doc = "The list of glob patterns to include."),
        @Param(
            name = "exclude",
            type = Sequence.class,
            generic1 = String.class,
            defaultValue = "[]",
            named = true,
            doc = "The list of glob patterns to exclude."),
        // TODO(bazel-team): accept booleans as well as integers? (and eventually migrate?)
        @Param(
            name = "exclude_directories",
            type = Integer.class,
            defaultValue = "1",
            named = true,
            doc = "A flag whether to exclude directories or not."),
        @Param(
            name = "allow_empty",
            type = Object.class,
            defaultValue = "unbound",
            named = true,
            doc =
                "Whether we allow glob patterns to match nothing. If `allow_empty` is False, each"
                    + " individual include pattern must match something and also the final"
                    + " result must be non-empty (after the matches of the `exclude` patterns are"
                    + " excluded).")
      },
      useLocation = true,
      useStarlarkThread = true)
  Sequence<?> glob(
      Sequence<?> include,
      Sequence<?> exclude,
      Integer excludeDirectories,
      Object allowEmpty,
      Location loc,
      StarlarkThread thread)
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "existing_rule",
      doc =
          "Returns a new mutable dict that describes the attributes of a rule instantiated in this "
              + "thread's package, or <code>None</code> if no rule instance of that name exists." //
              + "<p>The dict contains an entry for each attribute, except private ones, whose"
              + " names do not start with a letter. In addition, the dict contains entries for the"
              + " rule instance's <code>name</code> and <code>kind</code> (for example,"
              + " <code>'cc_binary'</code>)." //
              + "<p>The values of the dict represent attribute values as follows:" //
              + "<ul><li>Attributes of type str, int, and bool are represented as is.</li>" //
              + "<li>Labels are converted to strings of the form <code>':foo'</code> for targets"
              + " in the same package or <code>'//pkg:name'</code> for targets in a different"
              + " package.</li>" //
              + "<li>Lists are represented as tuples, and dicts are converted to new, mutable"
              + " dicts. Their elements are recursively converted in the same fashion.</li>" //
              + "<li><code>select</code> values are returned as is." //
              + "<li>Attributes for which no value was specified during rule instantiation and"
              + " whose default value is computed are excluded from the result. (Computed defaults"
              + " cannot be computed until the analysis phase.).</li>" //
              + "</ul>" //
              + "<p>If possible, avoid using this function. It makes BUILD files brittle and"
              + " order-dependent. Also, beware that it differs subtly from the two"
              + " other conversions of rule attribute values from internal form to Starlark: one"
              + " used by computed defaults, the other used by <code>ctx.attr.foo</code>.",
      parameters = {
        @Param(
            name = "name",
            type = String.class,
            // TODO(cparsons): This parameter should be positional-only.
            legacyNamed = true,
            doc = "The name of the target.")
      },
      useLocation = true,
      useStarlarkThread = true)
  Object existingRule(String name, Location loc, StarlarkThread thread)
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "existing_rules",
      doc =
          "Returns a new mutable dict describing the rules so far instantiated in this thread's"
              + " package. Each dict entry maps the name of the rule instance to the result that"
              + " would be returned by <code>existing_rule(name)</code>.<p><i>Note: If possible,"
              + " avoid using this function. It makes BUILD files brittle and order-dependent, and"
              + " it may be expensive especially if called within a loop.</i>",
      useLocation = true,
      useStarlarkThread = true)
  Dict<String, Dict<String, Object>> existingRules(Location loc, StarlarkThread thread)
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "package_group",
      doc =
          "This function defines a set of packages and assigns a label to the group. "
              + "The label can be referenced in <code>visibility</code> attributes.",
      parameters = {
        @Param(
            name = "name",
            type = String.class,
            named = true,
            positional = false,
            doc = "The unique name for this rule."),
        @Param(
            name = "packages",
            type = Sequence.class,
            generic1 = String.class,
            defaultValue = "[]",
            named = true,
            positional = false,
            doc = "A complete enumeration of packages in this group."),
        @Param(
            name = "includes",
            type = Sequence.class,
            generic1 = String.class,
            defaultValue = "[]",
            named = true,
            positional = false,
            doc = "Other package groups that are included in this one.")
      },
      useLocation = true,
      useStarlarkThread = true)
  NoneType packageGroup(
      String name, Sequence<?> packages, Sequence<?> includes, Location loc, StarlarkThread thread)
      throws EvalException;

  @SkylarkCallable(
      name = "exports_files",
      doc =
          "Specifies a list of files belonging to this package that are exported to other "
              + "packages but not otherwise mentioned.",
      parameters = {
        @Param(
            name = "srcs",
            type = Sequence.class,
            generic1 = String.class,
            named = true,
            doc = "The list of files to export."),
        // TODO(bazel-team): make it possible to express the precise type ListOf(LabelDesignator)
        @Param(
            name = "visibility",
            type = Sequence.class,
            defaultValue = "None",
            noneable = true,
            named = true,
            doc =
                "A visibility declaration can to be specified. The files will be visible to the "
                    + "targets specified. If no visibility is specified, the files will be visible "
                    + "to every package."),
        @Param(
            name = "licenses",
            type = Sequence.class,
            generic1 = String.class,
            noneable = true,
            named = true,
            defaultValue = "None",
            doc = "Licenses to be specified.")
      },
      useLocation = true,
      useStarlarkThread = true)
  NoneType exportsFiles(
      Sequence<?> srcs, Object visibility, Object licenses, Location loc, StarlarkThread thread)
      throws EvalException;

  @SkylarkCallable(
      name = "package_name",
      doc =
          "The name of the package being evaluated. "
              + "For example, in the BUILD file <code>some/package/BUILD</code>, its value "
              + "will be <code>some/package</code>. "
              + "If the BUILD file calls a function defined in a .bzl file, "
              + "<code>package_name()</code> will match the caller BUILD file package. "
              + "This function is equivalent to the deprecated variable <code>PACKAGE_NAME</code>.",
      parameters = {},
      useLocation = true,
      useStarlarkThread = true)
  String packageName(Location loc, StarlarkThread thread) throws EvalException;

  @SkylarkCallable(
      name = "repository_name",
      doc =
          "The name of the repository the rule or build extension is called from. "
              + "For example, in packages that are called into existence by the WORKSPACE stanza "
              + "<code>local_repository(name='local', path=...)</code> it will be set to "
              + "<code>@local</code>. In packages in the main repository, it will be set to "
              + "<code>@</code>. This function is equivalent to the deprecated variable "
              + "<code>REPOSITORY_NAME</code>.",
      parameters = {},
      useLocation = true,
      useStarlarkThread = true)
  String repositoryName(Location location, StarlarkThread thread) throws EvalException;
}
