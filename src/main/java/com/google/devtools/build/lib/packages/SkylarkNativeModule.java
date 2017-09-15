// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.syntax.BuiltinFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkSignatureProcessor;
import com.google.devtools.build.lib.syntax.Type.ConversionException;

/**
 * A class for the Skylark native module. TODO(laurentlb): Some definitions are duplicated from
 * PackageFactory.
 */
@SkylarkModule(
  name = "native",
  namespace = true,
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
public class SkylarkNativeModule {

  @SkylarkSignature(
    name = "glob",
    objectType = SkylarkNativeModule.class,
    returnType = SkylarkList.class,
    doc =
        "Glob returns a list of every file in the current package that:<ul>\n"
            + "<li>Matches at least one pattern in <code>include</code>.</li>\n"
            + "<li>Does not match any of the patterns in <code>exclude</code> "
            + "(default <code>[]</code>).</li></ul>\n"
            + "If the <code>exclude_directories</code> argument is enabled (set to <code>1</code>),"
            + " files of type directory will be omitted from the results (default <code>1</code>).",
    parameters = {
      @Param(
        name = "include",
        type = SkylarkList.class,
        generic1 = String.class,
        defaultValue = "[]",
        doc = "The list of glob patterns to include."
      ),
      @Param(
        name = "exclude",
        type = SkylarkList.class,
        generic1 = String.class,
        defaultValue = "[]",
        doc = "The list of glob patterns to exclude."
      ),
      // TODO(bazel-team): accept booleans as well as integers? (and eventually migrate?)
      @Param(
        name = "exclude_directories",
        type = Integer.class,
        defaultValue = "1",
        doc = "A flag whether to exclude directories or not."
      )
    },
    useAst = true,
    useEnvironment = true
  )
  private static final BuiltinFunction glob =
      new BuiltinFunction("glob") {
        public SkylarkList invoke(
            SkylarkList include,
            SkylarkList exclude,
            Integer excludeDirectories,
            FuncallExpression ast,
            Environment env)
            throws EvalException, ConversionException, InterruptedException {
          env.checkLoadingPhase("native.glob", ast.getLocation());
          return PackageFactory.callGlob(null, include, exclude, excludeDirectories != 0, ast, env);
        }
      };

  @SkylarkSignature(
    name = "existing_rule",
    objectType = SkylarkNativeModule.class,
    returnType = Object.class,
    doc =
        "Returns a dictionary representing the attributes of a previously defined rule, "
            + "or None if the rule does not exist.",
    parameters = {
      @Param(name = "name", type = String.class, doc = "The name of the rule.")
    },
    useAst = true,
    useEnvironment = true
  )
  private static final BuiltinFunction existingRule =
      new BuiltinFunction("existing_rule") {
        public Object invoke(String name, FuncallExpression ast, Environment env)
            throws EvalException, InterruptedException {
          env.checkLoadingOrWorkspacePhase("native.existing_rule", ast.getLocation());
          SkylarkDict<String, Object> rule = PackageFactory.callGetRuleFunction(name, ast, env);
          if (rule != null) {
            return rule;
          }

          return Runtime.NONE;
        }
      };

  /*
    If necessary, we could allow filtering by tag (anytag, alltags), name (regexp?), kind ?
    For now, we ignore this, since users can implement it in Skylark.
  */
  @SkylarkSignature(
    name = "existing_rules",
    objectType = SkylarkNativeModule.class,
    returnType = SkylarkDict.class,
    doc =
        "Returns a dict containing all the rules instantiated so far. "
            + "The map key is the name of the rule. The map value is equivalent to the "
            + "existing_rule output for that rule.",
    parameters = {},
    useAst = true,
    useEnvironment = true
  )
  private static final BuiltinFunction existingRules =
      new BuiltinFunction("existing_rules") {
        public SkylarkDict<String, SkylarkDict<String, Object>> invoke(
            FuncallExpression ast, Environment env)
            throws EvalException, InterruptedException {
          env.checkLoadingOrWorkspacePhase("native.existing_rules", ast.getLocation());
          return PackageFactory.callGetRulesFunction(ast, env);
        }
      };

  @SkylarkSignature(name = "package_group", objectType = SkylarkNativeModule.class,
      returnType = Runtime.NoneType.class,
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
  private static final BuiltinFunction packageGroup = new BuiltinFunction("package_group") {
      public Runtime.NoneType invoke(String name, SkylarkList packages, SkylarkList includes,
                FuncallExpression ast, Environment env) throws EvalException, ConversionException {
        env.checkLoadingPhase("native.package_group", ast.getLocation());
        return PackageFactory.callPackageFunction(name, packages, includes, ast, env);
      }
    };

  @SkylarkSignature(name = "exports_files", objectType = SkylarkNativeModule.class,
      returnType = Runtime.NoneType.class,
      doc = "Specifies a list of files belonging to this package that are exported to other "
          + "packages but not otherwise mentioned.",
      parameters = {
      @Param(name = "srcs", type = SkylarkList.class, generic1 = String.class,
          doc = "The list of files to export."),
      // TODO(bazel-team): make it possible to express the precise type ListOf(LabelDesignator)
      @Param(name = "visibility", type = SkylarkList.class, defaultValue = "None", noneable = true,
          doc = "A visibility declaration can to be specified. The files will be visible to the "
              + "targets specified. If no visibility is specified, the files will be visible to "
              + "every package."),
      @Param(name = "licenses", type = SkylarkList.class, generic1 = String.class, noneable = true,
          defaultValue = "None", doc = "Licenses to be specified.")},
      useAst = true, useEnvironment = true)
  private static final BuiltinFunction exportsFiles = new BuiltinFunction("exports_files") {
      public Runtime.NoneType invoke(SkylarkList srcs, Object visibility, Object licenses,
          FuncallExpression ast, Environment env)
          throws EvalException, ConversionException {
        env.checkLoadingPhase("native.exports_files", ast.getLocation());
        return PackageFactory.callExportsFiles(srcs, visibility, licenses, ast, env);
      }
    };

  @SkylarkSignature(
    name = "package_name",
    objectType = SkylarkNativeModule.class,
    returnType = String.class,
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
  private static final BuiltinFunction packageName =
      new BuiltinFunction("package_name") {
        public String invoke(FuncallExpression ast, Environment env)
            throws EvalException, ConversionException {
          env.checkLoadingPhase("native.package_name", ast.getLocation());
          return (String) env.lookup("PACKAGE_NAME");
        }
      };

  @SkylarkSignature(
    name = "repository_name",
    objectType = SkylarkNativeModule.class,
    returnType = String.class,
    doc =
        "The name of the repository the rule or build extension is called from. "
            + "For example, in packages that are called into existence by the WORKSPACE stanza "
            + "<code>local_repository(name='local', path=...)</code> it will be set to "
            + "<code>@local</code>. In packages in the main repository, it will be empty. This "
            + "function is equivalent to the deprecated variable <code>REPOSITORY_NAME</code>.",
    parameters = {},
    useAst = true,
    useEnvironment = true
  )
  private static final BuiltinFunction repositoryName =
      new BuiltinFunction("repository_name") {
        public String invoke(FuncallExpression ast, Environment env)
            throws EvalException, ConversionException {
          env.checkLoadingPhase("native.package_name", ast.getLocation());
          return (String) env.lookup("REPOSITORY_NAME");
        }
      };

  public static final SkylarkNativeModule NATIVE_MODULE = new SkylarkNativeModule();

  static {
    SkylarkSignatureProcessor.configureSkylarkFunctions(SkylarkNativeModule.class);
  }
}
