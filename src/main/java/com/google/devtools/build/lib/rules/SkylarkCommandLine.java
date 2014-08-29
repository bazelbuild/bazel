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

package com.google.devtools.build.lib.rules;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Function;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin.Param;
import com.google.devtools.build.lib.syntax.SkylarkFunction;
import com.google.devtools.build.lib.syntax.SkylarkFunction.SimpleSkylarkFunction;
import com.google.devtools.build.lib.syntax.SkylarkModule;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.util.LazyString;

import java.util.Map;

/**
 * A Skylark module class to create memory efficient command lines.
 */
@SkylarkModule(name = "Cmd", namespace = true,
    doc = "Module for creating memory efficient command lines.")
public class SkylarkCommandLine {

  @SkylarkBuiltin(name = "join_exec_paths",
      doc = "Creates a single command line argument joining the execution paths of a nested set "
          + "of files on the separator string.",
      objectType = SkylarkCommandLine.class,
      returnType = String.class,
      mandatoryParams = {
      @Param(name = "separator", doc = "the separator string to join on"),
      @Param(name = "files", type = SkylarkNestedSet.class, doc = "the files to concatenate")})
  private static SimpleSkylarkFunction joinExecPaths =
      new SimpleSkylarkFunction("join_exec_paths") {
    @Override
    public Object call(Map<String, Object> params, Location loc)
        throws EvalException {
      final String separator = cast(params.get("separator"), String.class, "separator", loc);
      final NestedSet<Artifact> artifacts =
          cast(params.get("files"), SkylarkNestedSet.class, "files", loc).getSet(Artifact.class);
      return new LazyString() {
        @Override
        public String toString() {
          return Artifact.joinExecPaths(separator, artifacts);
        }
      };
    }
  };

  public static final SkylarkCommandLine module = new SkylarkCommandLine();

  public static void registerFunctions(Environment env) {
    ImmutableList.Builder<Function> cmdFunctions = ImmutableList.builder();
    SkylarkFunction.collectSkylarkFunctionsFromFields(
        SkylarkCommandLine.class, null, cmdFunctions);
    for (Function fct : cmdFunctions.build()) {
      env.registerFunction(SkylarkCommandLine.class, fct.getName(), fct);
    }
  }

  public static void setupValidationEnvironment(
      Map<SkylarkType, Map<String, SkylarkType>> builtIn) {
    builtIn.get(SkylarkType.GLOBAL).put("Cmd", SkylarkType.of(SkylarkCommandLine.class));
    SkylarkFunction.collectSkylarkFunctionReturnTypesFromFields(
        SkylarkCommandLine.class, builtIn);
  }
}
