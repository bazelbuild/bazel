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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.auto.value.AutoBuilder;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableTable;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkRepositoryModule.RepositoryRuleFunction;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleFactory.InvalidRuleException;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.server.FailureDetails.ExternalDeps.Code;
import com.google.devtools.build.lib.skyframe.BzlLoadFunction;
import com.google.devtools.build.lib.skyframe.BzlLoadFunction.BzlLoadFailedException;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.Optional;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.spelling.SpellChecker;

/**
 * A fabricated module extension "innate" to each module, used to generate all repos defined using
 * {@code use_repo_rule}.
 */
final class InnateRunnableExtension implements RunnableExtension {
  private final ModuleKey moduleKey;
  private final ImmutableList<InnateExtensionRepo> repos;
  private final byte[] transitiveBzlDigest;
  private final BlazeDirectories directories;

  InnateRunnableExtension(
      ModuleKey moduleKey,
      ImmutableList<InnateExtensionRepo> repos,
      byte[] transitiveBzlDigest,
      BlazeDirectories directories) {
    this.moduleKey = moduleKey;
    this.repos = repos;
    this.transitiveBzlDigest = transitiveBzlDigest;
    this.directories = directories;
  }

  /** Returns null if a Skyframe restart is needed. */
  @Nullable
  static InnateRunnableExtension load(
      ModuleExtensionId extensionId,
      SingleExtensionUsagesValue usagesValue,
      StarlarkSemantics starlarkSemantics,
      Environment env,
      BlazeDirectories directories)
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
    ImmutableList<Tag> tags =
        Iterables.getOnlyElement(usagesValue.getExtensionUsages().values()).getTags();
    RepositoryMapping repoMapping = usagesValue.getRepoMappings().get(moduleKey);

    // Each tag of this usage defines a repo. The name of the tag is of the form
    // "<bzl_file_label>%<rule_name>". Collect the .bzl files referenced and load them.
    Label.RepoContext repoContext = Label.RepoContext.of(repoMapping.ownerRepo(), repoMapping);
    ArrayList<InnateExtensionRepo.Builder> repoBuilders = new ArrayList<>(tags.size());
    for (Tag tag : tags) {
      Iterator<String> parts = Splitter.on('%').split(tag.getTagName()).iterator();
      InnateExtensionRepo.Builder repoBuilder = InnateExtensionRepo.builder().tag(tag);
      repoBuilders.add(repoBuilder);
      try {
        Label label = Label.parseWithRepoContext(parts.next(), repoContext);
        BzlLoadFunction.checkValidLoadLabel(label, starlarkSemantics);
        repoBuilder.bzlLabel(label).ruleName(parts.next());
      } catch (LabelSyntaxException e) {
        throw ExternalDepsException.withCauseAndMessage(
            Code.BAD_MODULE, e, "bad repo rule .bzl file label at %s", tag.getLocation());
      }
    }
    ImmutableSet<BzlLoadValue.Key> loadKeys =
        repoBuilders.stream()
            .map(r -> BzlLoadValue.keyForBzlmod(r.bzlLabel()))
            .collect(toImmutableSet());
    HashSet<Label> digestedLabels = new HashSet<>();
    Fingerprint transitiveBzlDigest = new Fingerprint();
    SkyframeLookupResult loadResult = env.getValuesAndExceptions(loadKeys);
    for (InnateExtensionRepo.Builder repoBuilder : repoBuilders) {
      BzlLoadValue loadedBzl;
      try {
        loadedBzl =
            (BzlLoadValue)
                loadResult.getOrThrow(
                    BzlLoadValue.keyForBzlmod(repoBuilder.bzlLabel()),
                    BzlLoadFailedException.class);
      } catch (BzlLoadFailedException e) {
        throw ExternalDepsException.withCauseAndMessage(
            Code.BAD_MODULE,
            e,
            "error loading '%s' for repo rules, requested by %s",
            repoBuilder.bzlLabel(),
            repoBuilder.tag().getLocation());
      }
      if (loadedBzl == null) {
        return null;
      }
      repoBuilder.loadedBzl(loadedBzl);
      if (digestedLabels.add(repoBuilder.bzlLabel())) {
        // Only digest this BzlLoadValue if we haven't seen this bzl label before.
        transitiveBzlDigest.addBytes(loadedBzl.getTransitiveDigest());
      }
    }

    return new InnateRunnableExtension(
        moduleKey,
        repoBuilders.stream().map(InnateExtensionRepo.Builder::build).collect(toImmutableList()),
        transitiveBzlDigest.digestAndReset(),
        directories);
  }

  @Override
  public ModuleExtensionEvalFactors getEvalFactors() {
    return ModuleExtensionEvalFactors.create("", "");
  }

  @Override
  public byte[] getBzlTransitiveDigest() {
    return transitiveBzlDigest;
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
    var generatedRepoSpecs = ImmutableMap.<String, RepoSpec>builderWithExpectedSize(repos.size());
    // Instantiate the repos one by one.
    for (InnateExtensionRepo repo : repos) {
      Object exported = repo.loadedBzl().getModule().getGlobal(repo.ruleName());
      if (exported == null) {
        ImmutableSet<String> exportedRepoRules =
            repo.loadedBzl().getModule().getGlobals().entrySet().stream()
                .filter(e -> e.getValue() instanceof RepositoryRuleFunction)
                .map(Entry::getKey)
                .collect(toImmutableSet());
        throw ExternalDepsException.withMessage(
            Code.BAD_MODULE,
            "%s does not export a repository_rule called %s, yet its use is requested at" + " %s%s",
            repo.bzlLabel(),
            repo.ruleName(),
            repo.tag().getLocation(),
            SpellChecker.didYouMean(repo.ruleName(), exportedRepoRules));
      } else if (!(exported instanceof RepositoryRuleFunction)) {
        throw ExternalDepsException.withMessage(
            Code.BAD_MODULE,
            "%s exports a value called %s of type %s, yet a repository_rule is requested"
                + " at %s",
            repo.bzlLabel(),
            repo.ruleName(),
            Starlark.type(exported),
            repo.tag().getLocation());
      }
      RepositoryRuleFunction repoRule = (RepositoryRuleFunction) exported;
      Dict<String, Object> kwargs = repo.tag().getAttributeValues().attributes();
      // This cast should be safe since it should have been verified at tag creation time.
      String name = (String) kwargs.get("name");
      char separator =
          starlarkSemantics.getBool(BuildLanguageOptions.INCOMPATIBLE_USE_PLUS_IN_REPO_NAMES)
              ? '+'
              : '~';
      String prefixedName = usagesValue.getExtensionUniqueName() + separator + name;
      Rule ruleInstance;
      AttributeValues attributesValue;
      var fakeCallStackEntry =
          StarlarkThread.callStackEntry("InnateRunnableExtension.run", repo.tag().getLocation());
      // Rule creation strips the top-most entry from the call stack, so we need to add the fake
      // one twice.
      ImmutableList<StarlarkThread.CallStackEntry> fakeCallStack =
          ImmutableList.of(fakeCallStackEntry, fakeCallStackEntry);
      try {
        ruleInstance =
            BzlmodRepoRuleCreator.createRule(
                extensionId.getBzlFileLabel().getPackageIdentifier(),
                usagesValue.getRepoMappings().get(moduleKey),
                directories,
                starlarkSemantics,
                env.getListener(),
                fakeCallStack,
                repoRule.getRuleClass(),
                Maps.transformEntries(kwargs, (k, v) -> k.equals("name") ? prefixedName : v));
        attributesValue =
            AttributeValues.create(
                Maps.filterKeys(
                    Maps.transformEntries(kwargs, (k, v) -> ruleInstance.getAttr(k)),
                    k -> !k.equals("name")));
        AttributeValues.validateAttrs(
            attributesValue, String.format("%s '%s'", ruleInstance.getRuleClass(), name));
      } catch (InvalidRuleException | NoSuchPackageException | EvalException e) {
        throw ExternalDepsException.withCauseAndMessage(
            Code.BAD_MODULE,
            e,
            "error creating repo %s requested at %s",
            name,
            repo.tag().getLocation());
      }
      RepoSpec repoSpec =
          RepoSpec.builder()
              .setBzlFile(
                  repoRule
                      .getRuleClass()
                      .getRuleDefinitionEnvironmentLabel()
                      .getUnambiguousCanonicalForm())
              .setRuleClassName(repoRule.getRuleClass().getName())
              .setAttributes(attributesValue)
              .build();
      generatedRepoSpecs.put(name, repoSpec);
    }
    return new RunModuleExtensionResult(
        ImmutableMap.of(),
        ImmutableMap.of(),
        ImmutableMap.of(),
        generatedRepoSpecs.buildOrThrow(),
        Optional.of(ModuleExtensionMetadata.REPRODUCIBLE),
        ImmutableTable.of());
  }

  /** Information about a single repo to be created by an innate extension. */
  record InnateExtensionRepo(Label bzlLabel, String ruleName, Tag tag, BzlLoadValue loadedBzl) {
    static Builder builder() {
      return new AutoBuilder_InnateRunnableExtension_InnateExtensionRepo_Builder();
    }

    @AutoBuilder
    interface Builder {
      Builder bzlLabel(Label value);

      Label bzlLabel();

      Builder ruleName(String value);

      Builder tag(Tag value);

      Tag tag();

      Builder loadedBzl(BzlLoadValue value);

      InnateExtensionRepo build();
    }
  }
}
