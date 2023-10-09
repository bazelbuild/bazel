// Copyright 2021 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.StandardSystemProperty.OS_ARCH;
import static com.google.common.collect.ImmutableBiMap.toImmutableBiMap;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableTable;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.LockfileMode;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkRepositoryModule.RepositoryRuleFunction;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleFactory.InvalidRuleException;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.rules.repository.NeedsSkyframeRestartException;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.runtime.ProcessWrapper;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutor;
import com.google.devtools.build.lib.server.FailureDetails.ExternalDeps;
import com.google.devtools.build.lib.server.FailureDetails.ExternalDeps.Code;
import com.google.devtools.build.lib.skyframe.BzlLoadFunction;
import com.google.devtools.build.lib.skyframe.BzlLoadFunction.BzlLoadFailedException;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.function.Function;
import java.util.function.Supplier;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.spelling.SpellChecker;
import net.starlark.java.syntax.Location;

/**
 * Evaluates a single module extension. This function loads the .bzl file containing the extension,
 * runs its implementation function with a module_ctx object containing all relevant information,
 * and returns the generated repos.
 */
public class SingleExtensionEvalFunction implements SkyFunction {
  private final BlazeDirectories directories;
  private final Supplier<Map<String, String>> clientEnvironmentSupplier;
  private final DownloadManager downloadManager;

  private double timeoutScaling = 1.0;
  @Nullable private ProcessWrapper processWrapper = null;
  @Nullable private RepositoryRemoteExecutor repositoryRemoteExecutor = null;

  public SingleExtensionEvalFunction(
      BlazeDirectories directories,
      Supplier<Map<String, String>> clientEnvironmentSupplier,
      DownloadManager downloadManager) {
    this.directories = directories;
    this.clientEnvironmentSupplier = clientEnvironmentSupplier;
    this.downloadManager = downloadManager;
  }

  public void setTimeoutScaling(double timeoutScaling) {
    this.timeoutScaling = timeoutScaling;
  }

  public void setProcessWrapper(ProcessWrapper processWrapper) {
    this.processWrapper = processWrapper;
  }

  public void setRepositoryRemoteExecutor(RepositoryRemoteExecutor repositoryRemoteExecutor) {
    this.repositoryRemoteExecutor = repositoryRemoteExecutor;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SingleExtensionEvalFunctionException, InterruptedException {
    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }

    ModuleExtensionId extensionId = (ModuleExtensionId) skyKey.argument();
    SingleExtensionUsagesValue usagesValue =
        (SingleExtensionUsagesValue) env.getValue(SingleExtensionUsagesValue.key(extensionId));
    if (usagesValue == null) {
      return null;
    }
    RunnableExtension extension;
    if (extensionId.isInnate()) {
      extension = loadInnateRunnableExtension(extensionId, usagesValue, starlarkSemantics, env);
    } else {
      extension = loadRegularRunnableExtension(extensionId, usagesValue, starlarkSemantics, env);
    }
    if (extension == null) {
      return null;
    }

    // Check the lockfile first for that module extension
    LockfileMode lockfileMode = BazelLockFileFunction.LOCKFILE_MODE.get(env);
    if (!lockfileMode.equals(LockfileMode.OFF)) {
      BazelLockFileValue lockfile = (BazelLockFileValue) env.getValue(BazelLockFileValue.KEY);
      if (lockfile == null) {
        return null;
      }
      try {
        SingleExtensionEvalValue singleExtensionEvalValue =
            tryGettingValueFromLockFile(
                env,
                extensionId,
                extension.getEvalFactors(),
                extension.getEnvVars(),
                usagesValue,
                extension.getBzlTransitiveDigest(),
                lockfile);
        if (singleExtensionEvalValue != null) {
          return singleExtensionEvalValue;
        }
      } catch (NeedsSkyframeRestartException e) {
        return null;
      }
    }

    // Run that extension!
    RunModuleExtensionResult moduleExtensionResult =
        extension.run(env, usagesValue, starlarkSemantics, extensionId);
    if (moduleExtensionResult == null) {
      return null;
    }
    ImmutableMap<String, RepoSpec> generatedRepoSpecs =
        moduleExtensionResult.getGeneratedRepoSpecs();
    Optional<ModuleExtensionMetadata> moduleExtensionMetadata =
        moduleExtensionResult.getModuleExtensionMetadata();

    // At this point the extension has been evaluated successfully, but SingleExtensionEvalFunction
    // may still fail if imported repositories were not generated. However, since imports do not
    // influence the evaluation of the extension and the validation also runs when the extension
    // result is taken from the lockfile, we can already post the update event. This is necessary to
    // prevent the extension from rerunning when only the imports change.
    if (lockfileMode.equals(LockfileMode.UPDATE)) {
      env.getListener()
          .post(
              ModuleExtensionResolutionEvent.create(
                  extensionId,
                  extension.getEvalFactors(),
                  LockFileModuleExtension.builder()
                      .setBzlTransitiveDigest(extension.getBzlTransitiveDigest())
                      .setAccumulatedFileDigests(moduleExtensionResult.getAccumulatedFileDigests())
                      .setEnvVariables(extension.getEnvVars())
                      .setGeneratedRepoSpecs(generatedRepoSpecs)
                      .setModuleExtensionMetadata(moduleExtensionMetadata)
                      .build()));
    }
    return validateAndCreateSingleExtensionEvalValue(
        generatedRepoSpecs, moduleExtensionMetadata, extensionId, usagesValue, env);
  }

  /**
   * Tries to get the evaluation result from the lockfile, if it's still up-to-date. Otherwise,
   * returns {@code null}.
   *
   * @throws NeedsSkyframeRestartException in case we need a skyframe restart. Note that we
   *     <em>don't</em> return {@code null} in this case!
   */
  @Nullable
  private SingleExtensionEvalValue tryGettingValueFromLockFile(
      Environment env,
      ModuleExtensionId extensionId,
      ModuleExtensionEvalFactors extensionFactors,
      ImmutableMap<String, String> envVariables,
      SingleExtensionUsagesValue usagesValue,
      byte[] bzlTransitiveDigest,
      BazelLockFileValue lockfile)
      throws SingleExtensionEvalFunctionException,
          InterruptedException,
          NeedsSkyframeRestartException {
    LockfileMode lockfileMode = BazelLockFileFunction.LOCKFILE_MODE.get(env);

    var lockedExtensionMap = lockfile.getModuleExtensions().get(extensionId);
    LockFileModuleExtension lockedExtension =
        lockedExtensionMap == null ? null : lockedExtensionMap.get(extensionFactors);
    if (lockedExtension == null) {
      if (lockfileMode.equals(LockfileMode.ERROR)) {
        throw new SingleExtensionEvalFunctionException(
            ExternalDepsException.withMessage(
                Code.BAD_MODULE,
                "The module extension '%s'%s does not exist in the lockfile",
                extensionId,
                extensionFactors.isEmpty() ? "" : " for platform " + extensionFactors),
            Transience.PERSISTENT);
      }
      return null;
    }

    ImmutableMap<ModuleKey, ModuleExtensionUsage> lockedExtensionUsages;
    try {
      // TODO(salmasamy) might be nicer to precompute this table when we construct
      // BazelLockFileValue, without adding it to the json file
      ImmutableTable<ModuleExtensionId, ModuleKey, ModuleExtensionUsage> extensionUsagesById =
          BazelDepGraphFunction.getExtensionUsagesById(lockfile.getModuleDepGraph());
      lockedExtensionUsages = extensionUsagesById.row(extensionId);
    } catch (ExternalDepsException e) {
      throw new SingleExtensionEvalFunctionException(e, Transience.PERSISTENT);
    }

    boolean filesChanged = didFilesChange(env, lockedExtension.getAccumulatedFileDigests());
    // Check extension data in lockfile is still valid, disregarding usage information that is not
    // relevant for the evaluation of the extension.
    var trimmedLockedUsages = ModuleExtensionUsage.trimForEvaluation(lockedExtensionUsages);
    var trimmedUsages = ModuleExtensionUsage.trimForEvaluation(usagesValue.getExtensionUsages());
    if (!filesChanged
        && Arrays.equals(bzlTransitiveDigest, lockedExtension.getBzlTransitiveDigest())
        && trimmedUsages.equals(trimmedLockedUsages)
        && envVariables.equals(lockedExtension.getEnvVariables())) {
      return validateAndCreateSingleExtensionEvalValue(
          lockedExtension.getGeneratedRepoSpecs(),
          lockedExtension.getModuleExtensionMetadata(),
          extensionId,
          usagesValue,
          env);
    } else if (lockfileMode.equals(LockfileMode.ERROR)) {
      ImmutableList<String> extDiff =
          lockfile.getModuleExtensionDiff(
              extensionId,
              lockedExtension,
              bzlTransitiveDigest,
              filesChanged,
              envVariables,
              trimmedUsages,
              trimmedLockedUsages);
      throw new SingleExtensionEvalFunctionException(
          ExternalDepsException.withMessage(
              Code.BAD_MODULE,
              "Lock file is no longer up-to-date because: %s. "
                  + "Please run `bazel mod deps --lockfile_mode=update` to update your lockfile.",
              String.join(", ", extDiff)),
          Transience.PERSISTENT);
    }
    return null;
  }

  private boolean didFilesChange(
      Environment env, ImmutableMap<Label, String> accumulatedFileDigests)
      throws InterruptedException, NeedsSkyframeRestartException {
    // Turn labels into FileValue keys & get those values
    Map<Label, FileValue.Key> fileKeys = new HashMap<>();
    for (Label label : accumulatedFileDigests.keySet()) {
      try {
        RootedPath rootedPath = RepositoryFunction.getRootedPathFromLabel(label, env);
        fileKeys.put(label, FileValue.key(rootedPath));
      } catch (NeedsSkyframeRestartException e) {
        throw e;
      } catch (EvalException e) {
        // Consider those exception to be a cause for invalidation
        return true;
      }
    }
    SkyframeLookupResult result = env.getValuesAndExceptions(fileKeys.values());
    if (env.valuesMissing()) {
      throw new NeedsSkyframeRestartException();
    }

    // Compare the collected file values with the hashes stored in the lockfile
    for (Entry<Label, String> entry : accumulatedFileDigests.entrySet()) {
      FileValue fileValue = (FileValue) result.get(fileKeys.get(entry.getKey()));
      try {
        if (!entry.getValue().equals(RepositoryFunction.fileValueToMarkerValue(fileValue))) {
          return true;
        }
      } catch (IOException e) {
        // Consider those exception to be a cause for invalidation
        return true;
      }
    }
    return false;
  }

  /**
   * Validates the result of the module extension evaluation against the declared imports, throwing
   * an exception if validation fails, and returns a SingleExtensionEvalValue otherwise.
   *
   * <p>Since extension evaluation does not depend on the declared imports, the result of the
   * evaluation of the extension implementation function can be reused and persisted in the lockfile
   * even if validation fails.
   */
  private SingleExtensionEvalValue validateAndCreateSingleExtensionEvalValue(
      ImmutableMap<String, RepoSpec> generatedRepoSpecs,
      Optional<ModuleExtensionMetadata> moduleExtensionMetadata,
      ModuleExtensionId extensionId,
      SingleExtensionUsagesValue usagesValue,
      Environment env)
      throws SingleExtensionEvalFunctionException {
    // Evaluate the metadata before failing on invalid imports so that fixup warning are still
    // emitted in case of an error.
    if (moduleExtensionMetadata.isPresent()) {
      try {
        // TODO: ModuleExtensionMetadata#evaluate should throw ExternalDepsException instead of
        // EvalException.
        moduleExtensionMetadata
            .get()
            .evaluate(
                usagesValue.getExtensionUsages().values(),
                generatedRepoSpecs.keySet(),
                env.getListener());
      } catch (EvalException e) {
        env.getListener().handle(Event.error(e.getMessageWithStack()));
        throw new SingleExtensionEvalFunctionException(
            ExternalDepsException.withMessage(
                Code.BAD_MODULE,
                "error evaluating module extension %s in %s",
                extensionId.getExtensionName(),
                extensionId.getBzlFileLabel()),
            Transience.TRANSIENT);
      }
    }

    // Check that all imported repos have actually been generated.
    for (ModuleExtensionUsage usage : usagesValue.getExtensionUsages().values()) {
      for (Entry<String, String> repoImport : usage.getImports().entrySet()) {
        if (!generatedRepoSpecs.containsKey(repoImport.getValue())) {
          throw new SingleExtensionEvalFunctionException(
              ExternalDepsException.withMessage(
                  Code.BAD_MODULE,
                  "module extension \"%s\" from \"%s\" does not generate repository \"%s\", yet it"
                      + " is imported as \"%s\" in the usage at %s%s",
                  extensionId.getExtensionName(),
                  extensionId.getBzlFileLabel(),
                  repoImport.getValue(),
                  repoImport.getKey(),
                  usage.getLocation(),
                  SpellChecker.didYouMean(repoImport.getValue(), generatedRepoSpecs.keySet())),
              Transience.PERSISTENT);
        }
      }
    }

    return SingleExtensionEvalValue.create(
        generatedRepoSpecs,
        generatedRepoSpecs.keySet().stream()
            .collect(
                toImmutableBiMap(
                    e ->
                        RepositoryName.createUnvalidated(
                            usagesValue.getExtensionUniqueName() + "~" + e),
                    Function.identity())));
  }

  private BzlLoadValue loadBzlFile(
      Label bzlFileLabel,
      Location sampleUsageLocation,
      StarlarkSemantics starlarkSemantics,
      Environment env)
      throws SingleExtensionEvalFunctionException, InterruptedException {
    // Check that the .bzl label isn't crazy.
    try {
      BzlLoadFunction.checkValidLoadLabel(bzlFileLabel, starlarkSemantics);
    } catch (LabelSyntaxException e) {
      throw new SingleExtensionEvalFunctionException(
          ExternalDepsException.withCauseAndMessage(
              Code.BAD_MODULE, e, "invalid module extension label"),
          Transience.PERSISTENT);
    }

    // Load the .bzl file pointed to by the label.
    BzlLoadValue bzlLoadValue;
    try {
      bzlLoadValue =
          (BzlLoadValue)
              env.getValueOrThrow(
                  BzlLoadValue.keyForBzlmod(bzlFileLabel), BzlLoadFailedException.class);
    } catch (BzlLoadFailedException e) {
      throw new SingleExtensionEvalFunctionException(
          ExternalDepsException.withCauseAndMessage(
              Code.BAD_MODULE,
              e,
              "Error loading '%s' for module extensions, requested by %s: %s",
              bzlFileLabel,
              sampleUsageLocation,
              e.getMessage()),
          Transience.PERSISTENT);
    }
    return bzlLoadValue;
  }

  /**
   * An internal abstraction to support the two "flavors" of module extensions: the "regular", which
   * is declared using {@code module_extension} in a .bzl file; and the "innate", which is
   * fabricated from usages of {@code use_repo_rule} in MODULE.bazel files.
   *
   * <p>The general idiom is to "load" such a {@link RunnableExtension} object by getting as much
   * information about it as needed to determine whether it can be reused from the lockfile (hence
   * methods such as {@link #getEvalFactors()}, {@link #getBzlTransitiveDigest()}, {@link
   * #getEnvVars()}). Then the {@link #run} method can be called if it's determined that we can't
   * reuse the cached results in the lockfile and have to re-run this extension.
   */
  private interface RunnableExtension {
    ModuleExtensionEvalFactors getEvalFactors();

    byte[] getBzlTransitiveDigest();

    ImmutableMap<String, String> getEnvVars();

    @Nullable
    RunModuleExtensionResult run(
        Environment env,
        SingleExtensionUsagesValue usagesValue,
        StarlarkSemantics starlarkSemantics,
        ModuleExtensionId extensionId)
        throws InterruptedException, SingleExtensionEvalFunctionException;
  }

  /** Information about a single repo to be created by an innate extension. */
  @AutoValue
  abstract static class InnateExtensionRepo {
    abstract Label bzlLabel();

    abstract String ruleName();

    abstract Tag tag();

    abstract BzlLoadValue loadedBzl();

    static Builder builder() {
      return new AutoValue_SingleExtensionEvalFunction_InnateExtensionRepo.Builder();
    }

    @AutoValue.Builder
    abstract static class Builder {

      abstract Builder setBzlLabel(Label value);

      abstract Label bzlLabel();

      abstract Builder setRuleName(String value);

      abstract Builder setTag(Tag value);

      abstract Tag tag();

      abstract Builder setLoadedBzl(BzlLoadValue value);

      abstract InnateExtensionRepo build();
    }
  }

  @Nullable
  private InnateRunnableExtension loadInnateRunnableExtension(
      ModuleExtensionId extensionId,
      SingleExtensionUsagesValue usagesValue,
      StarlarkSemantics starlarkSemantics,
      Environment env)
      throws InterruptedException, SingleExtensionEvalFunctionException {
    // An innate extension should have a singular usage.
    if (usagesValue.getExtensionUsages().size() > 1) {
      throw new SingleExtensionEvalFunctionException(
          ExternalDepsException.withMessage(
              Code.BAD_MODULE,
              "innate module extension %s is used by multiple modules: %s",
              extensionId,
              usagesValue.getExtensionUsages().keySet()),
          Transience.PERSISTENT);
    }
    ModuleKey moduleKey = Iterables.getOnlyElement(usagesValue.getExtensionUsages().keySet());
    Preconditions.checkState(moduleKey.moduleFileLabel().equals(extensionId.getBzlFileLabel()));
    ImmutableList<Tag> tags =
        Iterables.getOnlyElement(usagesValue.getExtensionUsages().values()).getTags();
    RepositoryMapping repoMapping = usagesValue.getRepoMappings().get(moduleKey);

    // Each tag of this usage defines a repo. The name of the tag is of the form
    // "<bzl_file_label>%<rule_name>". Collect the .bzl files referenced and load them.
    Label.RepoContext repoContext =
        Label.RepoContext.of(moduleKey.getCanonicalRepoName(), repoMapping);
    ArrayList<InnateExtensionRepo.Builder> repoBuilders = new ArrayList<>(tags.size());
    for (Tag tag : tags) {
      Iterator<String> parts = Splitter.on('%').split(tag.getTagName()).iterator();
      InnateExtensionRepo.Builder repoBuilder = InnateExtensionRepo.builder().setTag(tag);
      repoBuilders.add(repoBuilder);
      try {
        Label label = Label.parseWithRepoContext(parts.next(), repoContext);
        BzlLoadFunction.checkValidLoadLabel(label, starlarkSemantics);
        repoBuilder.setBzlLabel(label).setRuleName(parts.next());
      } catch (LabelSyntaxException e) {
        throw new SingleExtensionEvalFunctionException(
            ExternalDepsException.withCauseAndMessage(
                Code.BAD_MODULE, e, "bad repo rule .bzl file label at %s", tag.getLocation()),
            Transience.PERSISTENT);
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
        throw new SingleExtensionEvalFunctionException(
            ExternalDepsException.withCauseAndMessage(
                Code.BAD_MODULE,
                e,
                "error loading '%s' for repo rules, requested by %s",
                repoBuilder.bzlLabel(),
                repoBuilder.tag().getLocation()),
            Transience.PERSISTENT);
      }
      if (loadedBzl == null) {
        return null;
      }
      repoBuilder.setLoadedBzl(loadedBzl);
      if (digestedLabels.add(repoBuilder.bzlLabel())) {
        // Only digest this BzlLoadValue if we haven't seen this bzl label before.
        transitiveBzlDigest.addBytes(loadedBzl.getTransitiveDigest());
      }
    }

    return new InnateRunnableExtension(
        moduleKey,
        repoBuilders.stream().map(InnateExtensionRepo.Builder::build).collect(toImmutableList()),
        transitiveBzlDigest.digestAndReset());
  }

  private final class InnateRunnableExtension implements RunnableExtension {
    private final ModuleKey moduleKey;
    private final ImmutableList<InnateExtensionRepo> repos;
    private final byte[] transitiveBzlDigest;

    InnateRunnableExtension(
        ModuleKey moduleKey, ImmutableList<InnateExtensionRepo> repos, byte[] transitiveBzlDigest) {
      this.moduleKey = moduleKey;
      this.repos = repos;
      this.transitiveBzlDigest = transitiveBzlDigest;
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
    public ImmutableMap<String, String> getEnvVars() {
      return ImmutableMap.of();
    }

    @Override
    public RunModuleExtensionResult run(
        Environment env,
        SingleExtensionUsagesValue usagesValue,
        StarlarkSemantics starlarkSemantics,
        ModuleExtensionId extensionId)
        throws InterruptedException, SingleExtensionEvalFunctionException {
      var generatedRepoSpecs = ImmutableMap.<String, RepoSpec>builderWithExpectedSize(repos.size());
      // Instiantiate the repos one by one.
      for (InnateExtensionRepo repo : repos) {
        Object exported = repo.loadedBzl().getModule().getGlobal(repo.ruleName());
        if (!(exported instanceof RepositoryRuleFunction)) {
          ImmutableSet<String> exportedRepoRules =
              repo.loadedBzl().getModule().getGlobals().entrySet().stream()
                  .filter(e -> e.getValue() instanceof RepositoryRuleFunction)
                  .map(Entry::getKey)
                  .collect(toImmutableSet());
          throw new SingleExtensionEvalFunctionException(
              ExternalDepsException.withMessage(
                  Code.BAD_MODULE,
                  "%s does not export a repository_rule called %s, yet its use is requested at"
                      + " %s%s",
                  repo.bzlLabel(),
                  repo.ruleName(),
                  repo.tag().getLocation(),
                  SpellChecker.didYouMean(repo.ruleName(), exportedRepoRules)),
              Transience.PERSISTENT);
        }
        RepositoryRuleFunction repoRule = (RepositoryRuleFunction) exported;
        Dict<String, Object> kwargs = repo.tag().getAttributeValues().attributes();
        // This cast should be safe since it should have been verified at tag creation time.
        String name = (String) kwargs.get("name");
        String prefixedName = usagesValue.getExtensionUniqueName() + "~" + name;
        Rule ruleInstance;
        try {
          ruleInstance =
              BzlmodRepoRuleCreator.createRule(
                  extensionId.getBzlFileLabel().getPackageIdentifier(),
                  usagesValue.getRepoMappings().get(moduleKey),
                  directories,
                  starlarkSemantics,
                  env.getListener(),
                  "SingleExtensionEval.createInnateExtensionRepoRule",
                  repoRule.getRuleClass(),
                  Maps.transformEntries(kwargs, (k, v) -> k.equals("name") ? prefixedName : v));
        } catch (InvalidRuleException | NoSuchPackageException | EvalException e) {
          throw new SingleExtensionEvalFunctionException(
              ExternalDepsException.withCauseAndMessage(
                  Code.BAD_MODULE,
                  e,
                  "error creating repo %s requested at %s",
                  name,
                  repo.tag().getLocation()),
              Transience.PERSISTENT);
        }
        RepoSpec repoSpec =
            RepoSpec.builder()
                .setBzlFile(
                    repoRule
                        .getRuleClass()
                        .getRuleDefinitionEnvironmentLabel()
                        .getUnambiguousCanonicalForm())
                .setRuleClassName(repoRule.getRuleClass().getName())
                .setAttributes(
                    AttributeValues.create(
                        Maps.transformEntries(kwargs, (k, v) -> ruleInstance.getAttr(k))))
                .build();
        generatedRepoSpecs.put(name, repoSpec);
      }
      return RunModuleExtensionResult.create(
          ImmutableMap.of(), generatedRepoSpecs.buildOrThrow(), Optional.empty());
    }
  }

  @Nullable
  private RegularRunnableExtension loadRegularRunnableExtension(
      ModuleExtensionId extensionId,
      SingleExtensionUsagesValue usagesValue,
      StarlarkSemantics starlarkSemantics,
      Environment env)
      throws InterruptedException, SingleExtensionEvalFunctionException {
    Location sampleUsageLocation =
        usagesValue.getExtensionUsages().values().iterator().next().getLocation();
    BzlLoadValue bzlLoadValue =
        loadBzlFile(extensionId.getBzlFileLabel(), sampleUsageLocation, starlarkSemantics, env);
    if (bzlLoadValue == null) {
      return null;
    }
    // TODO(wyv): Consider whether there's a need to check .bzl load visibility
    // (BzlLoadFunction#checkLoadVisibilities).
    // TODO(wyv): Consider refactoring to use PackageFunction#loadBzlModules, or the simpler API
    // that may be created by b/237658764.

    // Check that the .bzl file actually exports a module extension by our name.
    Object exported = bzlLoadValue.getModule().getGlobal(extensionId.getExtensionName());
    if (!(exported instanceof ModuleExtension)) {
      ImmutableSet<String> exportedExtensions =
          bzlLoadValue.getModule().getGlobals().entrySet().stream()
              .filter(e -> e.getValue() instanceof ModuleExtension)
              .map(Entry::getKey)
              .collect(toImmutableSet());
      throw new SingleExtensionEvalFunctionException(
          ExternalDepsException.withMessage(
              ExternalDeps.Code.BAD_MODULE,
              "%s does not export a module extension called %s, yet its use is requested at %s%s",
              extensionId.getBzlFileLabel(),
              extensionId.getExtensionName(),
              sampleUsageLocation,
              SpellChecker.didYouMean(extensionId.getExtensionName(), exportedExtensions)),
          Transience.PERSISTENT);
    }

    ModuleExtension extension = (ModuleExtension) exported;
    ImmutableMap<String, String> envVars =
        RepositoryFunction.getEnvVarValues(env, ImmutableSet.copyOf(extension.getEnvVariables()));
    if (envVars == null) {
      return null;
    }
    return new RegularRunnableExtension(
        BazelModuleContext.of(bzlLoadValue.getModule()), extension, envVars);
  }

  private final class RegularRunnableExtension implements RunnableExtension {
    private final BazelModuleContext bazelModuleContext;
    private final ModuleExtension extension;
    private final ImmutableMap<String, String> envVars;

    RegularRunnableExtension(
        BazelModuleContext bazelModuleContext,
        ModuleExtension extension,
        ImmutableMap<String, String> envVars) {
      this.bazelModuleContext = bazelModuleContext;
      this.extension = extension;
      this.envVars = envVars;
    }

    @Override
    public ModuleExtensionEvalFactors getEvalFactors() {
      return ModuleExtensionEvalFactors.create(
          extension.getOsDependent() ? OS.getCurrent().toString() : "",
          extension.getArchDependent() ? OS_ARCH.value() : "");
    }

    @Override
    public ImmutableMap<String, String> getEnvVars() {
      return envVars;
    }

    @Override
    public byte[] getBzlTransitiveDigest() {
      return bazelModuleContext.bzlTransitiveDigest();
    }

    @Nullable
    @Override
    public RunModuleExtensionResult run(
        Environment env,
        SingleExtensionUsagesValue usagesValue,
        StarlarkSemantics starlarkSemantics,
        ModuleExtensionId extensionId)
        throws InterruptedException, SingleExtensionEvalFunctionException {
      ModuleExtensionEvalStarlarkThreadContext threadContext =
          new ModuleExtensionEvalStarlarkThreadContext(
              usagesValue.getExtensionUniqueName() + "~",
              extensionId.getBzlFileLabel().getPackageIdentifier(),
              bazelModuleContext.repoMapping(),
              directories,
              env.getListener());
      ModuleExtensionContext moduleContext;
      Optional<ModuleExtensionMetadata> moduleExtensionMetadata;
      try (Mutability mu =
          Mutability.create("module extension", usagesValue.getExtensionUniqueName())) {
        StarlarkThread thread = new StarlarkThread(mu, starlarkSemantics);
        thread.setPrintHandler(Event.makeDebugPrintHandler(env.getListener()));
        moduleContext = createContext(env, usagesValue, starlarkSemantics, extensionId);
        threadContext.storeInThread(thread);
        try (SilentCloseable c =
            Profiler.instance()
                .profile(
                    ProfilerTask.BZLMOD,
                    () -> "evaluate module extension: " + extensionId.asTargetString())) {
          Object returnValue =
              Starlark.fastcall(
                  thread,
                  extension.getImplementation(),
                  new Object[] {moduleContext},
                  new Object[0]);
          if (returnValue != Starlark.NONE && !(returnValue instanceof ModuleExtensionMetadata)) {
            throw new SingleExtensionEvalFunctionException(
                ExternalDepsException.withMessage(
                    ExternalDeps.Code.BAD_MODULE,
                    "expected module extension %s in %s to return None or extension_metadata, got"
                        + " %s",
                    extensionId.getExtensionName(),
                    extensionId.getBzlFileLabel(),
                    Starlark.type(returnValue)),
                Transience.PERSISTENT);
          }
          if (returnValue instanceof ModuleExtensionMetadata) {
            moduleExtensionMetadata = Optional.of((ModuleExtensionMetadata) returnValue);
          } else {
            moduleExtensionMetadata = Optional.empty();
          }
        } catch (NeedsSkyframeRestartException e) {
          // Clean up and restart by returning null.
          try {
            if (moduleContext.getWorkingDirectory().exists()) {
              moduleContext.getWorkingDirectory().deleteTree();
            }
          } catch (IOException e1) {
            ExternalDepsException externalDepsException =
                ExternalDepsException.withCauseAndMessage(
                    ExternalDeps.Code.UNRECOGNIZED,
                    e1,
                    "Failed to clean up module context directory");
            throw new SingleExtensionEvalFunctionException(
                externalDepsException, Transience.TRANSIENT);
          }
          return null;
        } catch (EvalException e) {
          env.getListener().handle(Event.error(e.getMessageWithStack()));
          throw new SingleExtensionEvalFunctionException(
              ExternalDepsException.withMessage(
                  ExternalDeps.Code.BAD_MODULE,
                  "error evaluating module extension %s in %s",
                  extensionId.getExtensionName(),
                  extensionId.getBzlFileLabel()),
              Transience.TRANSIENT);
        }
      }
      return RunModuleExtensionResult.create(
          moduleContext.getAccumulatedFileDigests(),
          threadContext.getGeneratedRepoSpecs(),
          moduleExtensionMetadata);
    }

    private ModuleExtensionContext createContext(
        Environment env,
        SingleExtensionUsagesValue usagesValue,
        StarlarkSemantics starlarkSemantics,
        ModuleExtensionId extensionId)
        throws SingleExtensionEvalFunctionException {
      Path workingDirectory =
          directories
              .getOutputBase()
              .getRelative(LabelConstants.MODULE_EXTENSION_WORKING_DIRECTORY_LOCATION)
              .getRelative(usagesValue.getExtensionUniqueName());
      ArrayList<StarlarkBazelModule> modules = new ArrayList<>();
      for (AbridgedModule abridgedModule : usagesValue.getAbridgedModules()) {
        ModuleKey moduleKey = abridgedModule.getKey();
        try {
          modules.add(
              StarlarkBazelModule.create(
                  abridgedModule,
                  extension,
                  usagesValue.getRepoMappings().get(moduleKey),
                  usagesValue.getExtensionUsages().get(moduleKey)));
        } catch (ExternalDepsException e) {
          throw new SingleExtensionEvalFunctionException(e, Transience.PERSISTENT);
        }
      }
      ModuleExtensionUsage rootUsage = usagesValue.getExtensionUsages().get(ModuleKey.ROOT);
      boolean rootModuleHasNonDevDependency =
          rootUsage != null && rootUsage.getHasNonDevUseExtension();
      return new ModuleExtensionContext(
          workingDirectory,
          env,
          clientEnvironmentSupplier.get(),
          downloadManager,
          timeoutScaling,
          processWrapper,
          starlarkSemantics,
          repositoryRemoteExecutor,
          extensionId,
          StarlarkList.immutableCopyOf(modules),
          rootModuleHasNonDevDependency);
    }
  }

  static final class SingleExtensionEvalFunctionException extends SkyFunctionException {

    SingleExtensionEvalFunctionException(ExternalDepsException cause, Transience transience) {
      super(cause, transience);
    }
  }

  /* Holds the result data from running a module extension */
  @AutoValue
  abstract static class RunModuleExtensionResult {

    abstract ImmutableMap<Label, String> getAccumulatedFileDigests();

    abstract ImmutableMap<String, RepoSpec> getGeneratedRepoSpecs();

    abstract Optional<ModuleExtensionMetadata> getModuleExtensionMetadata();

    static RunModuleExtensionResult create(
        ImmutableMap<Label, String> accumulatedFileDigests,
        ImmutableMap<String, RepoSpec> generatedRepoSpecs,
        Optional<ModuleExtensionMetadata> moduleExtensionMetadata) {
      return new AutoValue_SingleExtensionEvalFunction_RunModuleExtensionResult(
          accumulatedFileDigests, generatedRepoSpecs, moduleExtensionMetadata);
    }
  }
}
