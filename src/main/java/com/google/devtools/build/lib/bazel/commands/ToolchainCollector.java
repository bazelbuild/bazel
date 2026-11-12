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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bazel.bzlmod.BazelDepGraphValue;
import com.google.devtools.build.lib.bazel.bzlmod.Module;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModuleRegistrations;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModuleRegistrations.RegisteredTarget;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModuleRegistrations.Registration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.Label.RepoContext;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
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
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

/**
 * Collects the toolchains that modules register, for {@code mod show_toolchains}.
 *
 * <p>The type of a toolchain is determined without a configuration, which is exact unless an alias
 * on the way selects its actual target based on the configuration.
 */
final class ToolchainCollector extends RegistrationCollector {

  private final ModuleKey baseModule;
  private final ImmutableList<String> toolchainTypes;
  private final Map<PackageIdentifier, Package> packages = new HashMap<>();

  /**
   * Creates a collector that only keeps toolchains of the given types, which are labels relative to
   * {@code baseModule}, or all toolchains if there are none.
   */
  ToolchainCollector(
      SkyframeExecutor skyframeExecutor,
      int threads,
      ExtendedEventHandler eventHandler,
      BazelDepGraphValue depGraph,
      ModuleKey baseModule,
      List<String> toolchainTypes) {
    super(skyframeExecutor, threads, eventHandler, depGraph);
    this.baseModule = baseModule;
    this.toolchainTypes = ImmutableList.copyOf(toolchainTypes);
  }

  /**
   * {@inheritDoc}
   *
   * <p>If there are toolchain types to keep, toolchains of other types are omitted, as are patterns
   * and unrequested modules that are left without toolchains or errors.
   *
   * @throws ResolutionException if a toolchain type to keep can't be resolved
   */
  @Override
  ImmutableList<ModuleRegistrations> collect(ImmutableSet<ModuleKey> modules)
      throws ResolutionException, InterruptedException {
    if (toolchainTypes.isEmpty()) {
      return super.collect(modules);
    }

    RepositoryName baseRepo = repo(baseModule);
    var mappingKey = RepositoryMappingValue.key(baseRepo);
    EvaluationResult<SkyValue> mapping = evaluate(ImmutableList.of(mappingKey));
    String mappingError = errorMessage(mapping.getError(mappingKey));
    if (mappingError != null) {
      throw new ResolutionException(mappingError);
    }
    var repoContext =
        RepoContext.of(
            baseRepo, ((RepositoryMappingValue) mapping.get(mappingKey)).repositoryMapping());
    var types = new HashSet<Label>();
    for (String type : toolchainTypes) {
      try {
        Label label = Label.parseWithRepoContext(type, repoContext);
        if (!label.getRepository().isVisible()) {
          throw new ResolutionException(
              String.format(
                  "No repository visible as '@%s' from %s",
                  label.getRepository().getName(),
                  label.getRepository().getContextRepoDisplayString()));
        }
        types.add(resolveAliases(label, "toolchain_type").getLabel());
      } catch (LabelSyntaxException | ResolutionException e) {
        throw new ResolutionException(
            String.format("In --toolchain_type %s option: %s", type, e.getMessage()));
      }
    }

    var result = ImmutableList.<ModuleRegistrations>builder();
    for (var module : super.collect(modules)) {
      var registrations = ImmutableList.<Registration>builder();
      for (var registration : module.registrations()) {
        switch (registration) {
          case Registration.Failed failed -> registrations.add(failed);
          case Registration.Expanded(String pattern, ImmutableList<RegisteredTarget> targets) -> {
            var matching =
                targets.stream()
                    .filter(
                        target ->
                            !(target instanceof RegisteredTarget.Toolchain toolchain)
                                || types.contains(toolchain.type()))
                    .collect(toImmutableList());
            if (!matching.isEmpty()) {
              registrations.add(new Registration.Expanded(pattern, matching));
            }
          }
        }
      }
      var filtered = new ModuleRegistrations(module.module(), registrations.build());
      if (!modules.isEmpty() || !filtered.registrations().isEmpty()) {
        result.add(filtered);
      }
    }
    return result.build();
  }

  @Override
  protected ImmutableList<String> patterns(Module module) {
    return module.getToolchainsToRegister();
  }

  @Override
  protected FilteringPolicy filteringPolicy() {
    return FilteringPolicies.ruleTypeExplicit(ToolchainRule.RULE_NAME);
  }

  @Override
  protected RegisteredTarget registeredTarget(Label label) throws InterruptedException {
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
    Package pkg = packages.get(packageId);
    if (pkg == null) {
      EvaluationResult<SkyValue> result = evaluate(ImmutableList.of(packageId));
      String error = errorMessage(result.getError(packageId));
      if (error != null) {
        throw new ResolutionException(error);
      }
      pkg = ((PackageValue) result.get(packageId)).getPackage();
      packages.put(packageId, pkg);
    }
    try {
      return pkg.getTarget(label.getName());
    } catch (NoSuchTargetException e) {
      throw new ResolutionException(e.getMessage());
    }
  }
}
