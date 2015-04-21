// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.GlobList;
import com.google.devtools.build.lib.syntax.SkylarkFunction;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkModule;
import com.google.devtools.build.lib.syntax.SkylarkSignature;
import com.google.devtools.build.lib.syntax.SkylarkSignature.Param;

import java.util.Map;

/**
 * A class for the Skylark native module.
 */
@SkylarkModule(name = "native", namespace = true, onlyLoadingPhase = true, doc =
      "A built-in module to support native rules and other package helper functions. "
    + "All native rules appear as functions in this module. Note that the native module is only "
    + "available in the loading phase (i.e. for macros, not for rule implementations).<br/>"
    + "Extra helper functions:")
public class SkylarkNativeModule {

  @SkylarkSignature(name = "glob", objectType = SkylarkNativeModule.class,
      returnType = GlobList.class,
      doc = "Glob returns a list of every file in the current package that:<ul>\n"
          + "<li>Matches at least one pattern in <code>include</code>.</li>\n"
          + "<li>Does not match any of the patterns in <code>exclude</code> "
          + "(default <code>[]</code>).</li></ul>\n"
          + "If the <code>exclude_directories</code> argument is enabled (set to <code>1</code>), "
          + "files of type directory will be omitted from the results (default <code>1</code>).",
      mandatoryPositionals = {
      @Param(name = "includes", type = SkylarkList.class, generic1 = String.class,
          defaultValue = "[]", doc = "The list of glob patterns to include.")},
      optionalPositionals = {
      @Param(name = "excludes", type = SkylarkList.class, generic1 = String.class,
          defaultValue = "[]", doc = "The list of glob patterns to exclude."),
      // TODO(bazel-team): accept booleans as well as integers? (and eventually migrate?)
      @Param(name = "exclude_directories", type = Integer.class, defaultValue = "1",
          doc = "A flag whether to exclude directories or not.")},
      useAst = true, useEnvironment = true)
  private static final SkylarkFunction glob = new SkylarkFunction("glob") {
      @Override
      public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
          throws EvalException, ConversionException, InterruptedException {
        return PackageFactory.callGlob(null, false, ast, env, new Object[] {
              kwargs.get("includes"),
              kwargs.get("excludes"),
              kwargs.get("exclude_directories")
            });
      }
    };

  @SkylarkSignature(name = "package_group", objectType = SkylarkNativeModule.class,
      returnType = Environment.NoneType.class,
      doc = "This function defines a set of packages and assigns a label to the group. "
          + "The label can be referenced in <code>visibility</code> attributes.",
      mandatoryNamedOnly = {
      @Param(name = "name", type = String.class,
          doc = "A unique name for this rule.")},
      optionalNamedOnly = {
      @Param(name = "packages", type = SkylarkList.class, generic1 = String.class,
          defaultValue = "[]",
          doc = "A complete enumeration of packages in this group."),
      @Param(name = "includes", type = SkylarkList.class, generic1 = String.class,
          defaultValue = "[]",
          doc = "Other package groups that are included in this one.")},
      useAst = true, useEnvironment = true)
  private static final SkylarkFunction packageGroup = new SkylarkFunction("package_group") {
    @Override
    public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
        throws EvalException, ConversionException {
      return PackageFactory.callPackageFunction(ast, env, new Object[] {
          kwargs.get("name"),
          kwargs.get("packages"),
          kwargs.get("includes")
      });
    }
  };

  @SkylarkSignature(name = "exports_files", objectType = SkylarkNativeModule.class,
      returnType = Environment.NoneType.class,
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
  private static final SkylarkFunction exportsFiles = new SkylarkFunction("exports_files") {
    @Override
    public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
        throws EvalException, ConversionException {
      return PackageFactory.callExportsFiles(ast, env, new Object[] {
          kwargs.get("srcs"),
          kwargs.get("visibility"),
          kwargs.get("licenses")
      });
    }
  };

  public static final SkylarkNativeModule NATIVE_MODULE = new SkylarkNativeModule();
}
