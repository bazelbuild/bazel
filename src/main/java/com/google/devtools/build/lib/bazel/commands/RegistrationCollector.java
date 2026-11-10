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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bazel.bzlmod.BazelDepGraphValue;
import com.google.devtools.build.lib.bazel.bzlmod.Module;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModuleRegistrations;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModuleRegistrations.RegisteredTarget;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModuleRegistrations.Registration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.SignedTargetPattern;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.rules.platform.ToolchainRule;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.TargetPatternValue;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternKey;
import com.google.devtools.build.lib.skyframe.toolchains.RegisteredExecutionPlatformsFunction;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Expands the toolchains or execution platforms that modules register, as {@code mod
 * show_toolchains} and {@code mod show_execution_platforms} display them.
 *
 * <p>Patterns are expanded with the same filters as in toolchain resolution. The type of a
 * toolchain is determined without a configuration, which is exact unless an alias on the way
 * selects its actual target based on the configuration.
 */
final class RegistrationCollector {

  /** The kind of registration to collect. */
  enum Kind {
    TOOLCHAINS,
    EXECUTION_PLATFORMS
  }

  private final SkyframeExecutor skyframeExecutor;
  private final int threads;
  private final ExtendedEventHandler eventHandler;
  private final BazelDepGraphValue depGraph;
  private final Kind kind;
  // Values are either a Package or the message of the error that occurred while loading it.
  private final Map<PackageIdentifier, Object> packages = new HashMap<>();

  RegistrationCollector(
      SkyframeExecutor skyframeExecutor,
      int threads,
      ExtendedEventHandler eventHandler,
      BazelDepGraphValue depGraph,
      Kind kind) {
    this.skyframeExecutor = skyframeExecutor;
    this.threads = threads;
    this.eventHandler = eventHandler;
    this.depGraph = depGraph;
    this.kind = kind;
  }

  /**
   * Returns the registrations of the given modules in the order in which Bazel considers them, or
   * those of all modules that register anything if {@code modules} is empty.
   *
   * <p>If {@code toolchainTypes} isn't empty, toolchains of other types are omitted, as are
   * patterns and unrequested modules that are left without toolchains or errors. The types must be
   * resolved with {@link #resolveToolchainType}.
   */
  ImmutableList<ModuleRegistrations> collect(
      ImmutableSet<ModuleKey> modules, ImmutableSet<Label> toolchainTypes)
      throws InterruptedException {
    var selected = new ArrayList<Module>();
    for (var module : depGraph.getDepGraph().values()) {
      if (modules.isEmpty() ? !patterns(module).isEmpty() : modules.contains(module.getKey())) {
        selected.add(module);
      }
    }

    EvaluationResult<SkyValue> mappings =
        evaluate(
            selected.stream().map(module -> RepositoryMappingValue.key(repo(module))).toList());
    var parsedPatterns = new HashMap<Module, List<Object>>();
    var patternKeys = new ArrayList<TargetPatternKey>();
    for (var module : selected) {
      var parsed = new ArrayList<Object>();
      var mappingKey = RepositoryMappingValue.key(repo(module));
      String mappingError = errorMessage(mappings.getError(mappingKey));
      for (String pattern : patterns(module)) {
        if (mappingError != null) {
          parsed.add(mappingError);
          continue;
        }
        var parser =
            new TargetPattern.Parser(
                PathFragment.EMPTY_FRAGMENT,
                repo(module),
                ((RepositoryMappingValue) mappings.get(mappingKey)).repositoryMapping());
        try {
          TargetPatternKey key =
              TargetPatternValue.key(
                  SignedTargetPattern.create(
                      parser.parse(pattern), SignedTargetPattern.Sign.POSITIVE),
                  filteringPolicy());
          parsed.add(key);
          patternKeys.add(key);
        } catch (TargetParsingException e) {
          parsed.add(e.getMessage());
        }
      }
      parsedPatterns.put(module, parsed);
    }

    EvaluationResult<SkyValue> expansions = evaluate(patternKeys);
    var result = ImmutableList.<ModuleRegistrations>builder();
    for (var module : selected) {
      var registrations = ImmutableList.<Registration>builder();
      List<Object> parsed = parsedPatterns.get(module);
      for (int i = 0; i < parsed.size(); i++) {
        String pattern = patterns(module).get(i);
        if (!(parsed.get(i) instanceof TargetPatternKey key)) {
          registrations.add(new Registration.Failed(pattern, (String) parsed.get(i)));
          continue;
        }
        String error = errorMessage(expansions.getError(key));
        if (error != null) {
          registrations.add(new Registration.Failed(pattern, error));
          continue;
        }
        var targets = ImmutableList.<RegisteredTarget>builder();
        for (var label : ((TargetPatternValue) expansions.get(key)).getTargets().getTargets()) {
          var target = registeredTarget(label);
          if (toolchainTypes.isEmpty()
              || !(target instanceof RegisteredTarget.Toolchain toolchain)
              || toolchainTypes.contains(toolchain.type())) {
            targets.add(target);
          }
        }
        var registration = new Registration.Expanded(pattern, targets.build());
        if (toolchainTypes.isEmpty() || !registration.targets().isEmpty()) {
          registrations.add(registration);
        }
      }
      var moduleRegistrations = new ModuleRegistrations(module.getKey(), registrations.build());
      if (!modules.isEmpty() || !moduleRegistrations.registrations().isEmpty()) {
        result.add(moduleRegistrations);
      }
    }
    return result.build();
  }

  /** Returns the label of the toolchain type that the given label refers to through aliases. */
  Label resolveToolchainType(Label label) throws ResolutionException, InterruptedException {
    return resolveAliases(label, "toolchain_type").getLabel();
  }

  private RegisteredTarget registeredTarget(Label label) throws InterruptedException {
    if (kind == Kind.EXECUTION_PLATFORMS) {
      return new RegisteredTarget.Platform(label);
    }
    try {
      Rule toolchain = resolveAliases(label, ToolchainRule.RULE_NAME);
      Label type =
          resolveAliases(
                  NonconfigurableAttributeMapper.of(toolchain)
                      .get(ToolchainRule.TOOLCHAIN_TYPE_ATTR, BuildType.LABEL),
                  "toolchain_type")
              .getLabel();
      RepositoryName ownerRepo = depGraph.getOwningModuleRepo(type.getRepository());
      return new RegisteredTarget.Toolchain(
          label, type, depGraph.getCanonicalRepoNameLookup().get(ownerRepo));
    } catch (ResolutionException e) {
      return new RegisteredTarget.Unresolved(label, e.getMessage());
    }
  }

  private Rule resolveAliases(Label label, String ruleClass)
      throws ResolutionException, InterruptedException {
    var seen = new HashSet<Label>();
    while (seen.add(label)) {
      var target = getTarget(label);
      if (target instanceof Rule rule) {
        if (rule.getRuleClass().equals(ruleClass)) {
          return rule;
        }
        if (rule.getRuleClass().equals("alias")) {
          var attributes = RawAttributeMapper.of(rule);
          if (attributes.isConfigurable("actual")) {
            throw new ResolutionException(
                String.format(
                    "%s selects its actual target based on the configuration",
                    label.getUnambiguousCanonicalForm()));
          }
          label = attributes.get("actual", BuildType.LABEL);
          continue;
        }
      }
      throw new ResolutionException(
          String.format(
              "%s is a %s, not a %s rule",
              label.getUnambiguousCanonicalForm(), target.getTargetKind(), ruleClass));
    }
    throw new ResolutionException(
        String.format("%s is part of an alias cycle", label.getUnambiguousCanonicalForm()));
  }

  private Target getTarget(Label label) throws ResolutionException, InterruptedException {
    var packageId = label.getPackageIdentifier();
    Object pkg = packages.get(packageId);
    if (pkg == null) {
      EvaluationResult<SkyValue> result = evaluate(ImmutableList.of(packageId));
      String error = errorMessage(result.getError(packageId));
      pkg = error != null ? error : ((PackageValue) result.get(packageId)).getPackage();
      packages.put(packageId, pkg);
    }
    if (pkg instanceof String error) {
      throw new ResolutionException(error);
    }
    try {
      return ((Package) pkg).getTarget(label.getName());
    } catch (NoSuchTargetException e) {
      throw new ResolutionException(e.getMessage());
    }
  }

  private EvaluationResult<SkyValue> evaluate(List<? extends SkyKey> keys)
      throws InterruptedException {
    return skyframeExecutor.evaluate(keys, /* keepGoing= */ true, threads, eventHandler);
  }

  private ImmutableList<String> patterns(Module module) {
    return switch (kind) {
      case TOOLCHAINS -> module.getToolchainsToRegister();
      case EXECUTION_PLATFORMS -> module.getExecutionPlatformsToRegister();
    };
  }

  private FilteringPolicy filteringPolicy() {
    return switch (kind) {
      case TOOLCHAINS -> FilteringPolicies.ruleTypeExplicit(ToolchainRule.RULE_NAME);
      case EXECUTION_PLATFORMS -> RegisteredExecutionPlatformsFunction.HAS_PLATFORM_INFO;
    };
  }

  private RepositoryName repo(Module module) {
    return depGraph.getCanonicalRepoNameLookup().inverse().get(module.getKey());
  }

  @Nullable
  private static String errorMessage(@Nullable ErrorInfo error) {
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
