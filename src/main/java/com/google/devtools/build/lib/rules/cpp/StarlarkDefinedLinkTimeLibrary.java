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

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext.LinkerInput;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.ExtraLinkTimeLibraryApi;
import java.util.HashMap;
import java.util.Objects;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.eval.Structure;
import net.starlark.java.eval.SymbolGenerator;
import net.starlark.java.eval.Tuple;

/**
 * An extra library to include in a link. The actual library is built at link time.
 *
 * <p>This can be used for non-C++ inputs to a C++ link. A class that implements this interface will
 * support transitively gathering all inputs from link dependencies, and then combine them all
 * together into a set of C++ libraries.
 *
 * <p>Any implementations must be immutable (and therefore thread-safe), because this is passed
 * between rules and accessed in a multi-threaded context.
 */
public class StarlarkDefinedLinkTimeLibrary implements ExtraLinkTimeLibraryApi, Structure {

  StarlarkDefinedLinkTimeLibrary(
      StarlarkCallable buildLibraryFunction, ImmutableMap<String, Object> objectMap) {
    this.buildLibraryFunction = buildLibraryFunction;
    this.objectMap = objectMap;
    this.key = Key.createKey(buildLibraryFunction, objectMap);
  }

  // Starlark function to create the output library.
  private final StarlarkCallable buildLibraryFunction;

  // Map of parameter names to values. Depsets should be combined when merging libraries.
  private final ImmutableMap<String, Object> objectMap;

  // Key object used to determine the "class" of the library implementation.
  // The equals method is used to determine equality.
  private final Key key;

  /**
   * Used to identify the "class" of this Library. The Java class is usually sufficient unless
   * behaviour is controlled dynamically.
   */
  public Key getKey() {
    return key;
  }

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
  public BuildLibraryOutput buildLibraries(
      RuleContext ruleContext,
      boolean staticMode,
      boolean forDynamicLibrary,
      SymbolGenerator<?> symbolGenerator)
      throws RuleErrorException, InterruptedException {
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

  @Nullable
  @Override
  public StarlarkValue getValue(String key) throws EvalException {
    return (StarlarkValue) objectMap.get(key);
  }

  @Override
  public ImmutableCollection<String> getFieldNames() {
    return objectMap.keySet();
  }

  @Override
  public void setField(String field, Object value) throws EvalException {
    throw Starlark.errorf("ExtraLinkLibrary does not support field assignment");
  }

  @Override
  public String getErrorMessageForUnknownField(String field) {
    return String.format("No argument '%s' was passed to this ExtraLinkLibrary", field);
  }

  /**
   * Class to identify the "class" of a StarlarkDefinedLinkTimeLibrary. Uses the build function and
   * the split between depset and non-depset parameters to determine equality.
   */
  private static class Key {

    private final Object builderFunction;
    private final ImmutableList<String> constantFields;
    private final ImmutableList<String> depsetFields;

    private Key(
        Object builderFunction,
        ImmutableList<String> constantFields,
        ImmutableList<String> depsetFields) {
      this.builderFunction = builderFunction;
      this.constantFields = constantFields;
      this.depsetFields = depsetFields;
    }

    public static Key createKey(Object builderFunction, ImmutableMap<String, Object> objectMap) {
      ImmutableList.Builder<String> depsetFields = ImmutableList.builder();
      ImmutableList.Builder<String> constantFields = ImmutableList.builder();
      for (String key : objectMap.keySet()) {
        if (objectMap.get(key) instanceof Depset) {
          depsetFields.add(key);
        } else {
          constantFields.add(key);
        }
      }
      return new Key(builderFunction, constantFields.build(), depsetFields.build());
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof Key key)) {
        return false;
      }
      return builderFunction.equals(key.builderFunction)
          && constantFields.equals(key.constantFields)
          && depsetFields.equals(key.depsetFields);
    }

    @Override
    public int hashCode() {
      return Objects.hash(builderFunction, constantFields, depsetFields);
    }
  }

  /** The Builder interface builds an ExtraLinkTimeLibrary. */
  /** Merge the ExtraLinkTimeLibrary based on the inputs. */
  public static StarlarkDefinedLinkTimeLibrary merge(
      ImmutableList<StarlarkDefinedLinkTimeLibrary> libraries) {
    StarlarkCallable buildLibraryFunction = null;
    HashMap<String, ImmutableList.Builder<Depset>> depsetMapBuilder = new HashMap<>();
    HashMap<String, Object> constantsMap = new HashMap<>();

    for (StarlarkDefinedLinkTimeLibrary library : libraries) {
      if (buildLibraryFunction == null) {
        buildLibraryFunction = library.buildLibraryFunction;
      }
      for (String key : library.objectMap.keySet()) {
        Object value = library.objectMap.get(key);
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
    return new StarlarkDefinedLinkTimeLibrary(buildLibraryFunction, builder.buildOrThrow());
  }
}
