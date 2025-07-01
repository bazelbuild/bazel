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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.rules.cpp.StarlarkDefinedLinkTimeLibrary.BuildLibraryOutput;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.eval.SymbolGenerator;
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

  private BuildLibraryOutput buildLibraries(
      RuleContext ruleContext,
      boolean staticMode,
      boolean forDynamicLibrary,
      SymbolGenerator<?> symbolGenerator)
      throws InterruptedException, RuleErrorException, EvalException {
    NestedSetBuilder<CcLinkingContext.LinkerInput> linkerInputs = NestedSetBuilder.linkOrder();
    NestedSetBuilder<Artifact> runtimeLibraries = NestedSetBuilder.linkOrder();
    for (StarlarkInfo extraLibrary : getExtraLibraries()) {
      BuildLibraryOutput buildLibraryOutput =
          StarlarkDefinedLinkTimeLibrary.buildLibraries(
              extraLibrary, ruleContext, staticMode, forDynamicLibrary, symbolGenerator);
      linkerInputs.addTransitive(buildLibraryOutput.getLinkerInputs());
      runtimeLibraries.addTransitive(buildLibraryOutput.getRuntimeLibraries());
    }
    return new BuildLibraryOutput(linkerInputs.build(), runtimeLibraries.build());
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
      throws EvalException, InterruptedException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    try {
      BuildLibraryOutput buildLibraryOutput =
          buildLibraries(
              starlarkRuleContext.getRuleContext(),
              staticMode,
              forDynamicLibrary,
              thread.getSymbolGenerator());
      Depset linkerInputs =
          Depset.of(CcLinkingContext.LinkerInput.class, buildLibraryOutput.getLinkerInputs());
      Depset runtimeLibraries = Depset.of(Artifact.class, buildLibraryOutput.getRuntimeLibraries());
      return Tuple.pair(linkerInputs, runtimeLibraries);
    } catch (RuleErrorException e) {
      throw new EvalException(e);
    }
  }
}
