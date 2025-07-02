// Copyright 2024 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableTable;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.bazel.repository.RepoRule;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkRepositoryModule.StarlarkRepoRule;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.packages.LabelConverter;
import com.google.devtools.build.lib.server.FailureDetails.ExternalDeps.Code;
import com.google.devtools.build.lib.skyframe.BzlLoadFailedException;
import com.google.devtools.build.lib.skyframe.BzlLoadFunction;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import java.util.Map.Entry;
import java.util.Optional;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.spelling.SpellChecker;
import net.starlark.java.syntax.Location;

/**
 * A fabricated module extension "innate" to each module, used to generate all repos defined using
 * {@code use_repo_rule} for a single repo rule.
 */
final class InnateRunnableExtension implements RunnableExtension {
  private final ModuleKey moduleKey;
  private final Label bzlLabel;
  private final String ruleName;
  private final BzlLoadValue loadedBzl;
  // Never empty.
  private final ImmutableList<Tag> tags;

  InnateRunnableExtension(
      ModuleKey moduleKey,
      Label bzlLabel,
      String ruleName,
      BzlLoadValue loadedBzl,
      ImmutableList<Tag> tags) {
    this.moduleKey = moduleKey;
    this.bzlLabel = bzlLabel;
    this.ruleName = ruleName;
    this.loadedBzl = loadedBzl;
    Preconditions.checkArgument(!tags.isEmpty());
    this.tags = tags;
  }

  /** Returns null if a Skyframe restart is needed. */
  @Nullable
  static InnateRunnableExtension load(
      ModuleExtensionId extensionId,
      SingleExtensionUsagesValue usagesValue,
      StarlarkSemantics starlarkSemantics,
      Environment env)
      throws InterruptedException, ExternalDepsException {
    // An innate extension should have a singular usage.
    if (usagesValue.getExtensionUsages().size() > 1) {
      throw ExternalDepsException.withMessage(
          Code.BAD_MODULE,
          "innate module extension %s is used by multiple modules: %s",
          extensionId,
          usagesValue.getExtensionUsages().keySet());
    }
    ModuleKey moduleKey = Iterables.getOnlyElement(usagesValue.getExtensionUsages().keySet());
    // ModuleFileFunction doesn't add usages for use_repo_rule without any instantiations, so we can
    // assume that there is at least one tag.
    ImmutableList<Tag> tags =
        Iterables.getOnlyElement(usagesValue.getExtensionUsages().values()).getTags();
    RepositoryMapping repoMapping = usagesValue.getRepoMappings().get(moduleKey);
    Label.RepoContext repoContext = Label.RepoContext.of(repoMapping.contextRepo(), repoMapping);

    // The name of the extension is of the form "<bzl_file_label> <rule_name>". Rule names cannot
    // contain spaces, so we can split on the last space.
    int lastSpace = extensionId.extensionName().lastIndexOf(' ');
    String rawLabel = extensionId.extensionName().substring(0, lastSpace);
    String ruleName = extensionId.extensionName().substring(lastSpace + 1);
    Location location = tags.getFirst().getLocation();
    Label bzlLabel;
    try {
      bzlLabel = Label.parseWithRepoContext(rawLabel, repoContext);
      BzlLoadFunction.checkValidLoadLabel(bzlLabel, starlarkSemantics);
    } catch (LabelSyntaxException e) {
      throw ExternalDepsException.withCauseAndMessage(
          Code.BAD_MODULE, e, "bad repo rule .bzl file label at %s", location);
    }
    if (ruleName.startsWith("_")) {
      throw ExternalDepsException.withMessage(
          Code.BAD_MODULE,
          "%s does not export a repository_rule called %s, yet its use is requested at %s",
          bzlLabel,
          ruleName,
          tags.getFirst().getLocation());
    }

    // Load the .bzl file.
    BzlLoadValue loadedBzl;
    try {
      loadedBzl =
          (BzlLoadValue)
              env.getValueOrThrow(
                  BzlLoadValue.keyForBzlmod(bzlLabel), BzlLoadFailedException.class);
    } catch (BzlLoadFailedException e) {
      throw ExternalDepsException.withCauseAndMessage(
          Code.BAD_MODULE,
          e,
          "error loading '%s' for repo rules, requested by %s",
          bzlLabel,
          location);
    }
    if (loadedBzl == null) {
      return null;
    }

    return new InnateRunnableExtension(moduleKey, bzlLabel, ruleName, loadedBzl, tags);
  }

  @Override
  public ModuleExtensionEvalFactors getEvalFactors() {
    return ModuleExtensionEvalFactors.create("", "");
  }

  @Override
  public byte[] getBzlTransitiveDigest() {
    return loadedBzl.getTransitiveDigest();
  }

  @Override
  public ImmutableMap<String, Optional<String>> getStaticEnvVars() {
    return ImmutableMap.of();
  }

  @Override
  public RunModuleExtensionResult run(
      Environment env,
      SingleExtensionUsagesValue usagesValue,
      StarlarkSemantics starlarkSemantics,
      ModuleExtensionId extensionId,
      RepositoryMapping mainRepositoryMapping)
      throws InterruptedException, ExternalDepsException {
    Object exported = loadedBzl.getModule().getGlobal(ruleName);
    if (exported == null) {
      ImmutableSet<String> exportedRepoRules =
          loadedBzl.getModule().getGlobals().entrySet().stream()
              .filter(e -> e.getValue() instanceof StarlarkRepoRule)
              .map(Entry::getKey)
              .collect(toImmutableSet());
      throw ExternalDepsException.withMessage(
          Code.BAD_MODULE,
          "%s does not export a repository_rule called %s, yet its use is requested at %s%s",
          bzlLabel,
          ruleName,
          tags.getFirst().getLocation(),
          SpellChecker.didYouMean(ruleName, exportedRepoRules));
    } else if (!(exported instanceof StarlarkRepoRule)) {
      throw ExternalDepsException.withMessage(
          Code.BAD_MODULE,
          "%s exports a value called %s of type %s, yet a repository_rule is requested at %s",
          bzlLabel,
          ruleName,
          Starlark.type(exported),
          tags.getFirst().getLocation());
    }
    RepoRule repoRule = ((StarlarkRepoRule) exported).getRepoRule();

    var generatedRepoSpecs = ImmutableMap.<String, RepoSpec>builderWithExpectedSize(tags.size());
    // Instantiate the repos one by one.
    for (Tag tag : tags) {
      Dict<String, Object> kwargs = tag.getAttributeValues().attributes();
      // This cast should be safe since it should have been verified at tag creation time.
      String name = (String) kwargs.get("name");
      ImmutableList<StarlarkThread.CallStackEntry> fakeCallStack =
          ImmutableList.of(StarlarkThread.callStackEntry("<toplevel>", tag.getLocation()));
      LabelConverter labelConverter =
          new LabelConverter(
              extensionId.bzlFileLabel().getPackageIdentifier(),
              usagesValue.getRepoMappings().get(moduleKey));
      generatedRepoSpecs.put(
          name,
          repoRule.instantiate(
              kwargs,
              fakeCallStack,
              labelConverter,
              env.getListener(),
              "to the %s".formatted(moduleKey.toDisplayString())));
    }
    return new RunModuleExtensionResult(
        ImmutableMap.of(),
        ImmutableMap.of(),
        ImmutableMap.of(),
        generatedRepoSpecs.buildOrThrow(),
        Optional.of(ModuleExtensionMetadata.REPRODUCIBLE),
        ImmutableTable.of());
  }
}
