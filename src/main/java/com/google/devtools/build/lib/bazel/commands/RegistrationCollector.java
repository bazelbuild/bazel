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
package com.google.devtools.build.lib.bazel.commands;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Table;
import com.google.devtools.build.lib.bazel.bzlmod.BazelDepGraphValue;
import com.google.devtools.build.lib.bazel.bzlmod.Module;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModuleRegistrations;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModuleRegistrations.RegisteredTarget;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModuleRegistrations.Registration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.SignedTargetPattern;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.TargetPatternValue;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternKey;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.ArrayList;
import javax.annotation.Nullable;

/**
 * Expands the target patterns that modules register for toolchain resolution, as {@code mod
 * show_toolchains} and {@code mod show_execution_platforms} display them.
 *
 * <p>Patterns are expanded with the same filters as in toolchain resolution.
 */
abstract sealed class RegistrationCollector permits ToolchainCollector, ExecutionPlatformCollector {

  private final SkyframeExecutor skyframeExecutor;
  private final int threads;
  private final ExtendedEventHandler eventHandler;
  protected final BazelDepGraphValue depGraph;

  RegistrationCollector(
      SkyframeExecutor skyframeExecutor,
      int threads,
      ExtendedEventHandler eventHandler,
      BazelDepGraphValue depGraph) {
    this.skyframeExecutor = skyframeExecutor;
    this.threads = threads;
    this.eventHandler = eventHandler;
    this.depGraph = depGraph;
  }

  /** Returns the patterns that the given module registers. */
  protected abstract ImmutableList<String> patterns(Module module);

  /** Returns the policy that selects the registered targets among those a pattern matches. */
  protected abstract FilteringPolicy filteringPolicy();

  /** Returns the given target that a registered pattern expands to. */
  protected abstract RegisteredTarget registeredTarget(Label label) throws InterruptedException;

  /**
   * Returns the registrations of the given modules in the order in which Bazel considers them, or
   * those of all modules that register anything if {@code modules} is empty.
   */
  ImmutableList<ModuleRegistrations> collect(ImmutableSet<ModuleKey> modules)
      throws ResolutionException, InterruptedException {
    var selected = new ArrayList<Module>();
    for (var module : depGraph.getDepGraph().values()) {
      if (modules.isEmpty() ? !patterns(module).isEmpty() : modules.contains(module.getKey())) {
        selected.add(module);
      }
    }

    EvaluationResult<SkyValue> mappings =
        evaluate(
            selected.stream()
                .map(module -> RepositoryMappingValue.key(repo(module.getKey())))
                .toList());
    Table<Module, String, TargetPatternKey> patternKeys = HashBasedTable.create();
    Table<Module, String, String> parseErrors = HashBasedTable.create();
    for (var module : selected) {
      var mappingKey = RepositoryMappingValue.key(repo(module.getKey()));
      String mappingError = errorMessage(mappings.getError(mappingKey));
      for (String pattern : patterns(module)) {
        if (mappingError != null) {
          parseErrors.put(module, pattern, mappingError);
          continue;
        }
        var parser =
            new TargetPattern.Parser(
                PathFragment.EMPTY_FRAGMENT,
                repo(module.getKey()),
                ((RepositoryMappingValue) mappings.get(mappingKey)).repositoryMapping());
        try {
          patternKeys.put(
              module,
              pattern,
              TargetPatternValue.key(
                  SignedTargetPattern.create(
                      parser.parse(pattern), SignedTargetPattern.Sign.POSITIVE),
                  filteringPolicy()));
        } catch (TargetParsingException e) {
          parseErrors.put(module, pattern, e.getMessage());
        }
      }
    }

    EvaluationResult<SkyValue> expansions = evaluate(patternKeys.values());
    var result = ImmutableList.<ModuleRegistrations>builder();
    for (var module : selected) {
      var registrations = ImmutableList.<Registration>builder();
      for (String pattern : patterns(module)) {
        String parseError = parseErrors.get(module, pattern);
        if (parseError != null) {
          registrations.add(new Registration.Failed(pattern, parseError));
          continue;
        }
        TargetPatternKey key = patternKeys.get(module, pattern);
        String error = errorMessage(expansions.getError(key));
        if (error != null) {
          registrations.add(new Registration.Failed(pattern, error));
          continue;
        }
        ResolvedTargets<Label> resolved = ((TargetPatternValue) expansions.get(key)).getTargets();
        if (resolved.hasError()) {
          registrations.add(
              new Registration.Failed(
                  pattern, "target pattern matched a package containing errors"));
          continue;
        }
        var targets = ImmutableList.<RegisteredTarget>builder();
        for (var label : resolved.getTargets()) {
          targets.add(registeredTarget(label));
        }
        registrations.add(new Registration.Expanded(pattern, targets.build()));
      }
      result.add(new ModuleRegistrations(module.getKey(), registrations.build()));
    }
    return result.build();
  }

  protected EvaluationResult<SkyValue> evaluate(Iterable<? extends SkyKey> keys)
      throws InterruptedException {
    return skyframeExecutor.evaluate(keys, /* keepGoing= */ true, threads, eventHandler);
  }

  protected RepositoryName repo(ModuleKey module) {
    return depGraph.getCanonicalRepoNameLookup().inverse().get(module);
  }

  @Nullable
  protected static String errorMessage(@Nullable ErrorInfo error) {
    if (error == null) {
      return null;
    }
    if (error.getException() != null) {
      return error.getException().getMessage();
    }
    return "cycle while evaluating " + error.getCycleInfo().getFirst().getTopKey();
  }

  /** Thrown when a label can't be resolved to a target of the expected rule class. */
  static final class ResolutionException extends Exception {
    ResolutionException(String message) {
      super(message);
    }
  }
}
