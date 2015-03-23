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
import com.google.devtools.build.lib.syntax.SkylarkBuiltin;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin.Param;
import com.google.devtools.build.lib.syntax.SkylarkFunction;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkModule;

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

  @SkylarkBuiltin(name = "glob", objectType = SkylarkNativeModule.class, 
      doc = "Glob returns a list of every file in the current package that:<ul>\n"
          + "<li>Matches at least one pattern in <code>include</code>.</li>\n"
          + "<li>Does not match any of the patterns in <code>exclude</code> "
          + "(default <code>[]</code>).</li></ul>\n"
          + "If the <code>exclude_directories</code> argument is enabled (set to <code>1</code>), "
          + "files of type directory will be omitted from the results (default <code>1</code>).",
      mandatoryParams = {
      @Param(name = "includes", type = SkylarkList.class, generic1 = String.class,
          doc = "The list of glob patterns to include.")},
      optionalParams = {
      @Param(name = "excludes", type = SkylarkList.class, generic1 = String.class,
          doc = "The list of glob patterns to exclude."),
      @Param(name = "exclude_directories", type = Integer.class,
          doc = "A flag whether to exclude directories or not.")})
  private static final SkylarkFunction glob = new SkylarkFunction("glob") {
    @Override
    public Object call(Map<String, Object> kwargs, FuncallExpression ast, Environment env)
        throws EvalException, ConversionException, InterruptedException {
      return PackageFactory.globCall(null, false, ast, env, new Object[] {
          kwargs.get("includes"),
          kwargs.get("excludes"),
          kwargs.get("exclude_directories")
      });
    }
  };

  public static final SkylarkNativeModule NATIVE_MODULE = new SkylarkNativeModule();
}
