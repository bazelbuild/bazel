// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext.LinkerInput;
import java.util.HashMap;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.SymbolGenerator;
import net.starlark.java.eval.Tuple;

/** Helper static methods for handling ExtraLinkTimeLibraries. */
public final class StarlarkDefinedLinkTimeLibrary {
  /** Output of {@link #buildLibraries}. Pair of libraries to link and runtime libraries. */
  static class BuildLibraryOutput {
    public NestedSet<LinkerInput> linkerInputs;
    public NestedSet<Artifact> runtimeLibraries;

    public BuildLibraryOutput(
        NestedSet<CcLinkingContext.LinkerInput> linkerInputs,
        NestedSet<Artifact> runtimeLibraries) {
      this.linkerInputs = linkerInputs;
      this.runtimeLibraries = runtimeLibraries;
    }

    public NestedSet<CcLinkingContext.LinkerInput> getLinkerInputs() {
      return linkerInputs;
    }

    public NestedSet<Artifact> getRuntimeLibraries() {
      return runtimeLibraries;
    }
  }

  /**
   * Build and return the LinkerInput inputs to pass to the C++ linker and the associated runtime
   * libraries.
   */
  public static BuildLibraryOutput buildLibraries(
      StarlarkInfo library,
      RuleContext ruleContext,
      boolean staticMode,
      boolean forDynamicLibrary,
      SymbolGenerator<?> symbolGenerator)
      throws RuleErrorException, InterruptedException, EvalException {

    StarlarkFunction buildLibraryFunction =
        library.getValue("build_library_func", StarlarkFunction.class);
    ImmutableMap.Builder<String, Object> objectMapBuilder = ImmutableMap.builder();
    for (String key : library.getFieldNames()) {
      if (key.equals("_key") || key.equals("build_library_func")) {
        continue;
      }
      objectMapBuilder.put(key, library.getValue(key));
    }
    Dict<String, Object> objectMap = Dict.immutableCopyOf(objectMapBuilder.buildOrThrow());

    ruleContext.initStarlarkRuleContext();
    StarlarkRuleContext starlarkContext = ruleContext.getStarlarkRuleContext();
    StarlarkSemantics semantics = starlarkContext.getStarlarkSemantics();

    Object response = null;
    try (Mutability mu = Mutability.create("extra_link_time_library_build_libraries_function")) {
      StarlarkThread thread =
          StarlarkThread.create(mu, semantics, "build_library_func callback", symbolGenerator);
      response =
          Starlark.call(
              thread,
              buildLibraryFunction,
              ImmutableList.of(starlarkContext, staticMode, forDynamicLibrary),
              objectMap);
    } catch (EvalException e) {
      throw new RuleErrorException(e);
    }
    String errorMsg =
        buildLibraryFunction.getName()
            + " in "
            + buildLibraryFunction.getLocation()
            + " should return (depset[CcLinkingContext], depset[File])";
    if (!(response instanceof Tuple responseTuple)) {
      throw new RuleErrorException(errorMsg);
    }
    if (responseTuple.size() != 2) {
      throw new RuleErrorException(errorMsg);
    }
    if (!(responseTuple.get(0) instanceof Depset) || !(responseTuple.get(1) instanceof Depset)) {
      throw new RuleErrorException(errorMsg);
    }
    try {
      return new BuildLibraryOutput(
          ((Depset) responseTuple.get(0)).getSet(CcLinkingContext.LinkerInput.class),
          ((Depset) responseTuple.get(1)).getSet(Artifact.class));
    } catch (Depset.TypeException e) {
      throw new RuleErrorException(e);
    }
  }

  /** The Builder interface builds an ExtraLinkTimeLibrary. */
  /** Merge the ExtraLinkTimeLibrary based on the inputs. */
  public static StarlarkInfo merge(ImmutableList<StarlarkInfo> libraries) {
    HashMap<String, ImmutableList.Builder<Depset>> depsetMapBuilder = new HashMap<>();
    HashMap<String, Object> constantsMap = new HashMap<>();

    for (StarlarkInfo library : libraries) {
      for (String key : library.getFieldNames()) {
        Object value = library.getValue(key);
        if (value instanceof Depset depset) {
          depsetMapBuilder.computeIfAbsent(key, k -> ImmutableList.builder()).add(depset);
        } else {
          constantsMap.put(key, value);
        }
      }
    }

    ImmutableMap.Builder<String, Object> builder = new ImmutableMap.Builder<>();
    for (String key : depsetMapBuilder.keySet()) {
      try {
        builder.put(
            key,
            Depset.fromDirectAndTransitive(
                Order.LINK_ORDER,
                ImmutableList.of(),
                depsetMapBuilder.get(key).build(),
                /* strict= */ true));
      } catch (EvalException e) {
        // should never happen; exception comes from bad order argument.
        throw new IllegalStateException(e);
      }
    }
    builder.putAll(constantsMap);
    // Note that we're returning Struct instead of the right provider. This situation will be
    // rectified once this code is rewritten to Starlark.
    return StructProvider.STRUCT.create(builder.buildOrThrow(), "");
  }

  private StarlarkDefinedLinkTimeLibrary() {}
}
