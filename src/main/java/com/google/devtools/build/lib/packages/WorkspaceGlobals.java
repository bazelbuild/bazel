// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import static net.starlark.java.eval.Starlark.NONE;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.RuleFactory.InvalidRuleException;
import com.google.devtools.build.lib.starlarkbuildapi.WorkspaceGlobalsApi;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/** A collection of global Starlark build API functions that apply to WORKSPACE files. */
public class WorkspaceGlobals implements WorkspaceGlobalsApi {

  private final boolean allowWorkspaceFunction;
  private final ImmutableMap<String, RuleClass> ruleClassMap;

  public WorkspaceGlobals(
      boolean allowWorkspaceFunction, ImmutableMap<String, RuleClass> ruleClassMap) {
    this.allowWorkspaceFunction = allowWorkspaceFunction;
    this.ruleClassMap = ruleClassMap;
  }

  @Override
  public void workspace(
      String name,
      StarlarkThread thread)
      throws EvalException, InterruptedException {
    if (!allowWorkspaceFunction) {
      throw Starlark.errorf(
          "workspace() function should be used only at the top of the WORKSPACE file");
    }
    RepositoryName.validateUserProvidedRepoName(name);
    String errorMessage = LabelValidator.validateTargetName(name);
    if (errorMessage != null) {
      throw Starlark.errorf("%s", errorMessage);
    }
    PackageFactory.getContext(thread).pkgBuilder.setWorkspaceName(name);
    Package.Builder builder = PackageFactory.getContext(thread).pkgBuilder;
    RuleClass localRepositoryRuleClass = ruleClassMap.get("local_repository");
    RuleClass bindRuleClass = ruleClassMap.get("bind");
    ImmutableMap<String, Object> kwargs = ImmutableMap.of("name", name, "path", ".");
    try {
      // This effectively adds a "local_repository(name = "<ws>", path = ".")"
      // definition to the WORKSPACE file.
      WorkspaceFactoryHelper.createAndAddRepositoryRule(
          builder,
          localRepositoryRuleClass,
          bindRuleClass,
          kwargs,
          thread.getCallStack());
    } catch (InvalidRuleException | NameConflictException | LabelSyntaxException e) {
      throw Starlark.errorf("%s", e.getMessage());
    }
    // Add entry in repository map from "@name" --> "@" to avoid issue where bazel
    // treats references to @name as a separate external repo
    builder.addRepositoryMappingEntry(RepositoryName.MAIN, name, RepositoryName.MAIN);
  }

  private static RepositoryName getRepositoryName(@Nullable Label label) {
    if (label == null) {
      // registration happened directly in the main WORKSPACE
      return RepositoryName.MAIN;
    }

    // registration happened in a loaded bzl file
    return label.getRepository();
  }

  private static ImmutableList<TargetPattern> parsePatterns(
      List<String> patterns, Package.Builder builder, StarlarkThread thread) throws EvalException {
    @Nullable // moduleContext is null if we're called directly from a WORKSPACE file.
    BazelModuleContext moduleContext =
        BazelModuleContext.of(Module.ofInnermostEnclosingStarlarkFunction(thread));
    RepositoryName myName =
        getRepositoryName((moduleContext != null ? moduleContext.label() : null));
    RepositoryMapping renaming = builder.getRepositoryMappingFor(myName);
    TargetPattern.Parser parser =
        new TargetPattern.Parser(PathFragment.EMPTY_FRAGMENT, myName, renaming);
    ImmutableList.Builder<TargetPattern> parsedPatterns = ImmutableList.builder();
    for (String pattern : patterns) {
      try {
        parsedPatterns.add(parser.parse(pattern));
      } catch (TargetParsingException e) {
        throw Starlark.errorf("error parsing target pattern \"%s\": %s", pattern, e.getMessage());
      }
    }
    return parsedPatterns.build();
  }

  @Override
  public void registerExecutionPlatforms(Sequence<?> platformLabels, StarlarkThread thread)
      throws EvalException {
    // Add to the package definition for later.
    Package.Builder builder = PackageFactory.getContext(thread).pkgBuilder;
    List<String> patterns = Sequence.cast(platformLabels, String.class, "platform_labels");
    builder.addRegisteredExecutionPlatforms(parsePatterns(patterns, builder, thread));
  }

  @Override
  public void registerToolchains(Sequence<?> toolchainLabels, StarlarkThread thread)
      throws EvalException {
    // Add to the package definition for later.
    Package.Builder builder = PackageFactory.getContext(thread).pkgBuilder;
    List<String> patterns = Sequence.cast(toolchainLabels, String.class, "toolchain_labels");
    builder.addRegisteredToolchains(parsePatterns(patterns, builder, thread));
  }

  @Override
  public void bind(String name, Object actual, StarlarkThread thread)
      throws EvalException, InterruptedException {
    Label nameLabel;
    try {
      nameLabel = Label.parseCanonical("//external:" + name);
    } catch (LabelSyntaxException e) {
      throw Starlark.errorf("%s", e.getMessage());
    }
    try {
      Package.Builder builder = PackageFactory.getContext(thread).pkgBuilder;
      RuleClass ruleClass = ruleClassMap.get("bind");
      WorkspaceFactoryHelper.addBindRule(
          builder,
          ruleClass,
          nameLabel,
          actual == NONE ? null : Label.parseCanonical((String) actual),
          thread.getCallStack());
    } catch (InvalidRuleException | Package.NameConflictException | LabelSyntaxException e) {
      throw Starlark.errorf("%s", e.getMessage());
    }
  }
}
