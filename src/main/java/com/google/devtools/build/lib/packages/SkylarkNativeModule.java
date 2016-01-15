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

import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature.Param;
import com.google.devtools.build.lib.syntax.BuiltinFunction;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkSignatureProcessor;
import com.google.devtools.build.lib.syntax.Type.ConversionException;

import java.util.Map;

/**
 * A class for the Skylark native module.
 */
@SkylarkModule(name = "native", namespace = true, doc =
    "A built-in module to support native rules and other package helper functions. "
    + "All native rules appear as functions in this module, e.g. <code>native.cc_library</code>. "
    + "Note that the native module is only available in the loading phase "
    + "(i.e. for macros, not for rule implementations). Attributes will ignore <code>None</code> "
    + "values, and treat them as if the attribute was unset.<br>"
    + "The following functions are also available:")
public class SkylarkNativeModule {

  // TODO(bazel-team): shouldn't we return a SkylarkList instead?
  @SkylarkSignature(
    name = "glob",
    objectType = SkylarkNativeModule.class,
    returnType = SkylarkList.class,
    doc =
        "Glob returns a list of every file in the current package that:<ul>\n"
            + "<li>Matches at least one pattern in <code>include</code>.</li>\n"
            + "<li>Does not match any of the patterns in <code>exclude</code> "
            + "(default <code>[]</code>).</li></ul>\n"
            + "If the <code>exclude_directories</code> argument is enabled (set to <code>1</code>), "
            + "files of type directory will be omitted from the results (default <code>1</code>).",
    mandatoryPositionals = {
      @Param(
        name = "include",
        type = SkylarkList.class,
        generic1 = String.class,
        defaultValue = "[]",
        doc = "The list of glob patterns to include."
      )
    },
    optionalPositionals = {
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
          return PackageFactory.callGlob(
              null, false, include, exclude, excludeDirectories != 0, ast, env);
        }
      };

  @SkylarkSignature(
    name = "rule",
    objectType = SkylarkNativeModule.class,
    returnType = Object.class,
    doc =
        "Returns a dictionary representing the attributes of a previously defined rule, "
            + "or None if the rule does not exist.",
    mandatoryPositionals = {
      @Param(name = "name", type = String.class, doc = "The name of the rule.")
    },
    useAst = true,
    useEnvironment = true
  )
  private static final BuiltinFunction getRule =
      new BuiltinFunction("rule") {
        public Object invoke(String name, FuncallExpression ast, Environment env)
            throws EvalException, InterruptedException {
          env.checkLoadingPhase("native.rule", ast.getLocation());
          Map<String, Object> rule = PackageFactory.callGetRuleFunction(name, ast, env);
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
    name = "rules",
    objectType = SkylarkNativeModule.class,
    returnType = Map.class,
    doc =
        "Returns a dict containing all the rules instantiated so far. "
            + "The map key is the name of the rule. The map value is equivalent to the "
            + "get_rule output for that rule.",
    mandatoryPositionals = {},
    useAst = true,
    useEnvironment = true
  )
  private static final BuiltinFunction getRules =
      new BuiltinFunction("rules") {
        public Map invoke(FuncallExpression ast, Environment env)
            throws EvalException, InterruptedException {
          env.checkLoadingPhase("native.rules", ast.getLocation());
          return PackageFactory.callGetRulesFunction(ast, env);
        }
      };

  @SkylarkSignature(name = "package_group", objectType = SkylarkNativeModule.class,
      returnType = Runtime.NoneType.class,
      doc = "This function defines a set of packages and assigns a label to the group. "
          + "The label can be referenced in <code>visibility</code> attributes.",
      mandatoryNamedOnly = {
      @Param(name = "name", type = String.class,
          doc = "The unique name for this rule.")},
      optionalNamedOnly = {
      @Param(name = "packages", type = SkylarkList.class, generic1 = String.class,
          defaultValue = "[]",
          doc = "A complete enumeration of packages in this group."),
      @Param(name = "includes", type = SkylarkList.class, generic1 = String.class,
          defaultValue = "[]",
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
      mandatoryPositionals = {
      @Param(name = "srcs", type = SkylarkList.class, generic1 = String.class,
          doc = "The list of files to export.")},
      optionalPositionals = {
      // TODO(bazel-team): make it possible to express the precise type ListOf(LabelDesignator)
      @Param(name = "visibility", type = SkylarkList.class,
          noneable = true,
          doc = "A visibility declaration can to be specified. The files will be visible to the "
              + "targets specified. If no visibility is specified, the files will be visible to "
              + "every package."),
      @Param(name = "licenses", type = SkylarkList.class, generic1 = String.class, noneable = true,
          doc = "Licenses to be specified.")},
      useAst = true, useEnvironment = true)
  private static final BuiltinFunction exportsFiles = new BuiltinFunction("exports_files") {
      public Runtime.NoneType invoke(SkylarkList srcs, Object visibility, Object licenses,
          FuncallExpression ast, Environment env)
          throws EvalException, ConversionException {
        env.checkLoadingPhase("native.exports_files", ast.getLocation());
        return PackageFactory.callExportsFiles(srcs, visibility, licenses, ast, env);
      }
    };

  public static final SkylarkNativeModule NATIVE_MODULE = new SkylarkNativeModule();

  static {
    SkylarkSignatureProcessor.configureSkylarkFunctions(SkylarkNativeModule.class);
  }
}
