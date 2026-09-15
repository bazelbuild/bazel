// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue.RepositoryMappingResolutionException;
import com.google.devtools.build.lib.util.ResourceConverter;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFloat;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.SymbolGenerator;

/** Loads and calls the function specified by {@code --local_resources_function}. */
final class LocalResourcesFunction {
  private LocalResourcesFunction() {}

  static ImmutableMap<String, Double> load(CommandEnvironment env, @Nullable String reference)
      throws InvalidConfigurationException, InterruptedException {
    if (reference == null || reference.isEmpty()) {
      return ImmutableMap.of();
    }

    String context = "Invalid --local_resources_function='" + reference + "': ";
    int delimiter = reference.indexOf('%');
    if (delimiter <= 0
        || delimiter == reference.length() - 1
        || reference.indexOf('%', delimiter + 1) >= 0) {
      throw new InvalidConfigurationException(
          context + "expected //pkg:file.bzl%function or @repo//pkg:file.bzl%function");
    }

    var executor = env.getSkyframeExecutor();
    Label label;
    try {
      label =
          Label.parseWithRepoContext(
              reference.substring(0, delimiter),
              Label.RepoContext.of(
                  RepositoryName.MAIN, executor.getMainRepoMapping(env.getReporter())));
    } catch (LabelSyntaxException | RepositoryMappingResolutionException e) {
      throw new InvalidConfigurationException(context + e.getMessage(), e);
    }
    if (!label.getName().endsWith(".bzl")) {
      throw new InvalidConfigurationException(context + "the label must refer to a .bzl file");
    }

    BzlLoadValue.Key key = BzlLoadValue.keyForBuild(label);
    EvaluationResult<SkyValue> result =
        executor.evaluateSkyKeys(env.getReporter(), ImmutableList.of(key), /* keepGoing= */ false);
    if (result.hasError()) {
      ErrorInfo error = result.getError(key);
      executor.getCyclesReporter().reportCycles(error.getCycleInfo(), key, env.getReporter());
      Throwable cause = error.getException();
      throw new InvalidConfigurationException(
          context + "failed to load " + label + (cause == null ? "" : ": " + cause.getMessage()),
          cause);
    }

    BzlLoadValue value = (BzlLoadValue) result.get(key);
    String functionName = reference.substring(delimiter + 1);
    Object symbol = value.getModule().getGlobal(functionName);
    if (symbol == null) {
      throw new InvalidConfigurationException(
          context + "function '" + functionName + "' not found in " + label);
    }
    if (!(symbol instanceof StarlarkFunction function)) {
      throw new InvalidConfigurationException(
          context
              + "'"
              + functionName
              + "' must be a Starlark function, got "
              + Starlark.type(symbol));
    }

    // Call once per build, before execution strategies and the resource manager are configured.
    // Skyframe tracks the .bzl file and its transitive loads, including repository-generated files.
    try (Mutability mutability = Mutability.create("local_resources_function")) {
      StarlarkThread thread =
          StarlarkThread.create(
              mutability,
              executor.getEffectiveStarlarkSemantics(
                  env.getOptions().getOptions(BuildLanguageOptions.class)),
              "local_resources_function",
              SymbolGenerator.createTransient());
      thread.setPrintHandler(Event.makeDebugPrintHandler(env.getReporter()));
      Dict<String, Object> resources =
          Dict.cast(
              Starlark.positionalOnlyCall(thread, function),
              String.class,
              Object.class,
              "local resources");
      ImmutableMap.Builder<String, Double> parsedResources = ImmutableMap.builder();
      var converter = new ResourceConverter.AssignmentConverter();
      for (Map.Entry<String, Object> resource : resources.entrySet()) {
        String name = resource.getKey();
        Object amount = resource.getValue();
        if (name.isEmpty() || name.contains("=")) {
          throw Starlark.errorf("invalid local resource name '%s'", name);
        }
        if (!(amount instanceof StarlarkInt
            || amount instanceof StarlarkFloat
            || amount instanceof String)) {
          throw Starlark.errorf(
              "local resource '%s' must be an int, float, or string, got %s",
              name, Starlark.type(amount));
        }
        double capacity;
        try {
          capacity = converter.convert(name + "=" + amount).getValue();
        } catch (OptionsParsingException e) {
          throw Starlark.errorf("invalid value for local resource '%s': %s", name, e.getMessage());
        }
        if (!Double.isFinite(capacity) || capacity < 0) {
          throw Starlark.errorf("local resource '%s' must be finite and nonnegative", name);
        }
        parsedResources.put(name, capacity);
      }
      return parsedResources.buildOrThrow();
    } catch (EvalException e) {
      throw new InvalidConfigurationException(context + e.getMessageWithStack(), e);
    }
  }
}
