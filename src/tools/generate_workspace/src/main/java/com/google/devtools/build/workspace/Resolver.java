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

package com.google.devtools.build.workspace;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.bazel.BazelMain;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory.EnvironmentExtension;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.WorkspaceFactory;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.workspace.maven.DefaultModelResolver;
import com.google.devtools.build.workspace.maven.Rule;

import org.apache.maven.model.building.ModelSource;
import org.apache.maven.model.resolution.UnresolvableModelException;

import java.io.IOException;
import java.util.List;

/**
 * Finds the transitive dependencies of a WORKSPACE file.
 */
public class Resolver {

  private final RuleClassProvider ruleClassProvider;
  private final ImmutableList<EnvironmentExtension> environmentExtensions;
  private final EventHandler handler;
  private final com.google.devtools.build.workspace.maven.Resolver resolver;

  Resolver(com.google.devtools.build.workspace.maven.Resolver resolver, EventHandler handler) {
    this.resolver = resolver;
    this.handler = handler;
    ConfiguredRuleClassProvider.Builder ruleClassBuilder =
        new ConfiguredRuleClassProvider.Builder();
    List<BlazeModule> blazeModules = BlazeRuntime.createModules(BazelMain.BAZEL_MODULES);
    ImmutableList.Builder<EnvironmentExtension> environmentExtensions = ImmutableList.builder();
    for (BlazeModule blazeModule : blazeModules) {
      blazeModule.initializeRuleClasses(ruleClassBuilder);
      environmentExtensions.add(blazeModule.getPackageEnvironmentExtension());
    }
    this.ruleClassProvider = ruleClassBuilder.build();
    this.environmentExtensions = environmentExtensions.build();
  }

  /**
   * Converts the WORKSPACE file content into an ExternalPackage.
   */
  public Package parse(Path workspacePath) {
    resolver.addHeader(workspacePath.getPathString());
    Package.LegacyBuilder builder =
        Package.newExternalPackageBuilder(workspacePath, ruleClassProvider.getRunfilesPrefix());
    try (Mutability mutability = Mutability.create("External Package %s", workspacePath)) {
      new WorkspaceFactory(builder, ruleClassProvider, environmentExtensions, mutability)
          .parse(ParserInputSource.create(workspacePath));
    } catch (IOException | InterruptedException e) {
      handler.handle(Event.error(Location.fromFile(workspacePath), e.getMessage()));
    }

    return builder.build();
  }

  /**
   * Calculates transitive dependencies of the given //external package.
   */
  public void resolveTransitiveDependencies(Package externalPackage) {
    Location location = Location.fromFile(externalPackage.getFilename());
    for (Target target : externalPackage.getTargets()) {
      // Targets are //external:foo.
      if (target.getTargetKind().startsWith("maven_jar ")) {
        RepositoryName repositoryName;
        try {
          repositoryName = RepositoryName.create("@" + target.getName());
        } catch (LabelSyntaxException e) {
          handler.handle(Event.error(location, "Invalid repository name for " + target + ": "
              + e.getMessage()));
          return;
        }
        com.google.devtools.build.lib.packages.Rule workspaceRule =
            externalPackage.getRule(repositoryName.strippedName());

        DefaultModelResolver modelResolver = resolver.getModelResolver();
        AttributeMap attributeMap = AggregatingAttributeMapper.of(workspaceRule);
        Rule rule;
        try {
          rule = new Rule(attributeMap.get("artifact", Type.STRING));
        } catch (Rule.InvalidRuleException e) {
          handler.handle(Event.error(location, "Couldn't get attribute: " + e.getMessage()));
          return;
        }
        if (attributeMap.isAttributeValueExplicitlySpecified("repository")) {
          modelResolver.addUserRepository(attributeMap.get("repository", Type.STRING));
          rule.setRepository(attributeMap.get("repository", Type.STRING), handler);
        }
        if (attributeMap.isAttributeValueExplicitlySpecified("sha1")) {
          rule.setSha1(attributeMap.get("sha1", Type.STRING));
        } else {
          rule.setSha1(resolver.downloadSha1(rule));
        }

        ModelSource modelSource;
        try {
          modelSource = modelResolver.resolveModel(
              rule.groupId(), rule.artifactId(), rule.version());
        } catch (UnresolvableModelException e) {
          handler.handle(Event.error(
              "Could not resolve model for " + target + ": " + e.getMessage()));
          continue;
        }
        resolver.addRootDependency(rule);
        resolver.resolveEffectiveModel(modelSource, Sets.<String>newHashSet(), rule);
      } else if (!target.getTargetKind().startsWith("bind")
          && !target.getTargetKind().startsWith("source ")) {
        handler.handle(Event.warn(location, "Cannot fetch transitive dependencies for " + target
            + " yet, skipping"));
      }
    }
  }
}
