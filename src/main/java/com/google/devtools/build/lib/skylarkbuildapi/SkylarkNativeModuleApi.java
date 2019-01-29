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
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;

/**
 * Interface for a module with native rule and package helper functions.
 */
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
            + "The following functions are also available:"
)
public interface SkylarkNativeModuleApi {

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
            type = SkylarkList.class,
            generic1 = String.class,
            defaultValue = "[]",
            legacyNamed = true,
            doc = "The list of glob patterns to include."),
        @Param(
            name = "exclude",
            type = SkylarkList.class,
            generic1 = String.class,
            defaultValue = "[]",
            legacyNamed = true,
            doc = "The list of glob patterns to exclude."),
        // TODO(bazel-team): accept booleans as well as integers? (and eventually migrate?)
        @Param(
            name = "exclude_directories",
            type = Integer.class,
            defaultValue = "1",
            legacyNamed = true,
            doc = "A flag whether to exclude directories or not.")
      },
      useAst = true,
      useEnvironment = true)
  public SkylarkList<?> glob(
      SkylarkList<?> include,
      SkylarkList<?> exclude,
      Integer excludeDirectories,
      FuncallExpression ast,
      Environment env)
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "existing_rule",
      doc =
          "Returns a dictionary representing the attributes of a previously defined target, or "
              + "<code>None</code> if the target does not exist."
              + ""
              + "<p><i>Note: If possible, avoid using this function. It makes BUILD files brittle "
              + "and order-dependent.</i>",
      parameters = {
        @Param(
            name = "name",
            type = String.class,
            legacyNamed = true,
            doc = "The name of the target.")
      },
      useAst = true,
      useEnvironment = true)
  public Object existingRule(String name, FuncallExpression ast, Environment env)
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "existing_rules",
      doc =
          "Returns a dictionary containing all the targets instantiated so far. The map key is the "
              + "name of the target. The map value is equivalent to the <code>existing_rule</code> "
              + "output for that target."
              + ""
              + "<p><i>Note: If possible, avoid using this function. It makes BUILD files brittle "
              + "and order-dependent.</i>",
      useAst = true,
      useEnvironment = true)
  public SkylarkDict<String, SkylarkDict<String, Object>> existingRules(
      FuncallExpression ast, Environment env) throws EvalException, InterruptedException;

  @SkylarkCallable(name = "package_group",
      doc = "This function defines a set of packages and assigns a label to the group. "
          + "The label can be referenced in <code>visibility</code> attributes.",
      parameters = {
      @Param(name = "name", type = String.class, named = true, positional = false,
          doc = "The unique name for this rule."),
      @Param(name = "packages", type = SkylarkList.class, generic1 = String.class,
          defaultValue = "[]", named = true, positional = false,
          doc = "A complete enumeration of packages in this group."),
      @Param(name = "includes", type = SkylarkList.class, generic1 = String.class,
          defaultValue = "[]", named = true, positional = false,
          doc = "Other package groups that are included in this one.")},
      useAst = true, useEnvironment = true)
  public Runtime.NoneType packageGroup(String name, SkylarkList<?> packages,
      SkylarkList<?> includes,
      FuncallExpression ast, Environment env) throws EvalException;

  @SkylarkCallable(
      name = "exports_files",
      doc =
          "Specifies a list of files belonging to this package that are exported to other "
              + "packages but not otherwise mentioned.",
      parameters = {
        @Param(
            name = "srcs",
            type = SkylarkList.class,
            generic1 = String.class,
            legacyNamed = true,
            doc = "The list of files to export."),
        // TODO(bazel-team): make it possible to express the precise type ListOf(LabelDesignator)
        @Param(
            name = "visibility",
            type = SkylarkList.class,
            defaultValue = "None",
            noneable = true,
            legacyNamed = true,
            doc =
                "A visibility declaration can to be specified. The files will be visible to the "
                    + "targets specified. If no visibility is specified, the files will be visible "
                    + "to every package."),
        @Param(
            name = "licenses",
            type = SkylarkList.class,
            generic1 = String.class,
            noneable = true,
            legacyNamed = true,
            defaultValue = "None",
            doc = "Licenses to be specified.")
      },
      useAst = true,
      useEnvironment = true)
  public Runtime.NoneType exportsFiles(
      SkylarkList<?> srcs,
      Object visibility,
      Object licenses,
      FuncallExpression ast,
      Environment env)
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
    useAst = true,
    useEnvironment = true
  )
  public String packageName(FuncallExpression ast, Environment env)
      throws EvalException;

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
    useEnvironment = true
  )
  public String repositoryName(Location location, Environment env)
      throws EvalException;
}
