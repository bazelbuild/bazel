// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.Depset.TypeException;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.eval.Tuple;

/**
 * A list of extra libraries to include in a link. These are non-C++ libraries that are built from
 * inputs gathered from all the dependencies. The dependencies have no way to coordinate, so each
 * one will add an ExtraLinkTimeLibrary to its CcLinkParams. ExtraLinkTimeLibrary is an interface,
 * and all ExtraLinkTimeLibrary objects of the same class will be gathered together.
 */
public final class ExtraLinkTimeLibraries implements StarlarkValue {

  static final ExtraLinkTimeLibraries EMPTY = new ExtraLinkTimeLibraries(ImmutableList.of());

  /**
   * We can have multiple different kinds of lists of libraries to include at link time. We map from
   * the class type to an actual instance.
   */
  private final Collection<StarlarkInfo> extraLibraries;

  private ExtraLinkTimeLibraries(Collection<StarlarkInfo> extraLibraries) {
    this.extraLibraries = extraLibraries;
  }

  /** Creates an instance from a single library. */
  public static ExtraLinkTimeLibraries of(StarlarkInfo library) {
    return new ExtraLinkTimeLibraries(ImmutableList.of(library));
  }

  /** Return the set of extra libraries. */
  public Collection<StarlarkInfo> getExtraLibraries() {
    return extraLibraries;
  }

  /** Get the set of extra libraries for Starlark. */
  @StarlarkMethod(name = "extra_libraries", documented = false, useStarlarkThread = true)
  public Sequence<StarlarkInfo> getExtraLibrariesForStarlark(StarlarkThread thread)
      throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return StarlarkList.immutableCopyOf(getExtraLibraries());
  }

  /** Merges a collection of {@link ExtraLinkTimeLibraries}. */
  public static ExtraLinkTimeLibraries merge(
      Iterable<ExtraLinkTimeLibraries> extraLinkTimeLibrariesCollection) {
    Map<Object, ImmutableList.Builder<StarlarkInfo>> libraries = new LinkedHashMap<>();
    for (ExtraLinkTimeLibraries extraLinkTimeLibraries : extraLinkTimeLibrariesCollection) {
      for (StarlarkInfo extraLinkTimeLibrary : extraLinkTimeLibraries.getExtraLibraries()) {
        Object key = extraLinkTimeLibrary.getValue("_key");
        libraries.computeIfAbsent(key, k -> ImmutableList.builder()).add(extraLinkTimeLibrary);
      }
    }
    if (libraries.isEmpty()) {
      return EMPTY;
    }
    ImmutableList.Builder<StarlarkInfo> extraLibraries = ImmutableList.builder();
    for (ImmutableList.Builder<StarlarkInfo> builder : libraries.values()) {
      extraLibraries.add(StarlarkDefinedLinkTimeLibrary.merge(builder.build()));
    }
    return new ExtraLinkTimeLibraries(extraLibraries.build());
  }

  @StarlarkMethod(
      name = "build_libraries",
      documented = false,
      useStarlarkThread = true,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
        @Param(name = "static_mode", positional = false, named = true),
        @Param(name = "for_dynamic_library", positional = false, named = true),
      })
  public Tuple getBuildLibrariesForStarlark(
      StarlarkRuleContext starlarkRuleContext,
      boolean staticMode,
      boolean forDynamicLibrary,
      StarlarkThread thread)
      throws EvalException, InterruptedException, TypeException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);

    NestedSetBuilder<StarlarkInfo> linkerInputs = NestedSetBuilder.linkOrder();
    NestedSetBuilder<Artifact> runtimeLibraries = NestedSetBuilder.linkOrder();
    for (StarlarkInfo extraLibrary : getExtraLibraries()) {
      StarlarkFunction buildLibraryFunction =
          extraLibrary.getValue("build_library_func", StarlarkFunction.class);
      ImmutableMap.Builder<String, Object> objectMapBuilder = ImmutableMap.builder();
      for (String key : extraLibrary.getFieldNames()) {
        if (key.equals("_key") || key.equals("build_library_func")) {
          continue;
        }
        objectMapBuilder.put(key, extraLibrary.getValue(key));
      }
      Dict<String, Object> objectMap = Dict.immutableCopyOf(objectMapBuilder.buildOrThrow());
      Object response =
          Starlark.call(
              thread,
              buildLibraryFunction,
              ImmutableList.of(starlarkRuleContext, staticMode, forDynamicLibrary),
              objectMap);
      if (!(response instanceof Tuple responseTuple)
          || responseTuple.size() != 2
          || !(responseTuple.get(0) instanceof Depset)
          || !(responseTuple.get(1) instanceof Depset)) {
        throw new EvalException(
            buildLibraryFunction.getName()
                + " in "
                + buildLibraryFunction.getLocation()
                + " should return (depset[CcLinkingContext], depset[File])");
      }
      linkerInputs.addTransitive(((Depset) responseTuple.get(0)).getSet(StarlarkInfo.class));
      runtimeLibraries.addTransitive(((Depset) responseTuple.get(1)).getSet(Artifact.class));
    }
    Depset linkerInputsDepset = Depset.of(StarlarkInfo.class, linkerInputs.build());
    Depset runtimeLibrariesDepset = Depset.of(Artifact.class, runtimeLibraries.build());
    return Tuple.of(linkerInputsDepset, runtimeLibrariesDepset);
  }
}
