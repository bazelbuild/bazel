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

import static com.google.common.collect.ImmutableBiMap.toImmutableBiMap;
import static java.util.stream.Collectors.joining;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.bazel.bzlmod.RunnableExtension.RunModuleExtensionResult;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.LockfileMode;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.bazel.repository.starlark.NeedsSkyframeRestartException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.rules.repository.RepoRecordedInput;
import com.google.devtools.build.lib.runtime.ProcessWrapper;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutor;
import com.google.devtools.build.lib.server.FailureDetails.ExternalDeps.Code;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.function.Function;
import java.util.function.Supplier;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * Evaluates a single module extension. This function loads the .bzl file containing the extension,
 * runs its implementation function with a module_ctx object containing all relevant information,
 * and returns the generated repos.
 */
public class SingleExtensionEvalFunction implements SkyFunction {
  private final BlazeDirectories directories;
  private final Supplier<ImmutableMap<String, String>> repoEnvSupplier;
  private final Supplier<ImmutableMap<String, String>> nonstrictRepoEnvSupplier;

  private double timeoutScaling = 1.0;
  @Nullable private ProcessWrapper processWrapper = null;
  @Nullable private RepositoryRemoteExecutor repositoryRemoteExecutor = null;
  @Nullable private DownloadManager downloadManager = null;

  public SingleExtensionEvalFunction(
      BlazeDirectories directories,
      Supplier<ImmutableMap<String, String>> repoEnvSupplier,
      Supplier<ImmutableMap<String, String>> nonstrictRepoEnvSupplier) {
    this.directories = directories;
    this.repoEnvSupplier = repoEnvSupplier;
    this.nonstrictRepoEnvSupplier = nonstrictRepoEnvSupplier;
  }

  public void setDownloadManager(DownloadManager downloadManager) {
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
    RepositoryMappingValue mainRepoMappingValue =
        (RepositoryMappingValue) env.getValue(RepositoryMappingValue.key(RepositoryName.MAIN));
    if (mainRepoMappingValue == null) {
      return null;
    }

    ModuleExtensionId extensionId = (ModuleExtensionId) skyKey.argument();
    SingleExtensionUsagesValue usagesValue =
        (SingleExtensionUsagesValue) env.getValue(SingleExtensionUsagesValue.key(extensionId));
    if (usagesValue == null) {
      return null;
    }
    RunnableExtension extension;
    try {
      if (extensionId.isInnate()) {
        extension = InnateRunnableExtension.load(extensionId, usagesValue, starlarkSemantics, env);
      } else {
        extension =
            RegularRunnableExtension.load(
                extensionId,
                usagesValue,
                starlarkSemantics,
                env,
                directories,
                repoEnvSupplier.get(),
                nonstrictRepoEnvSupplier.get(),
                timeoutScaling,
                processWrapper,
                repositoryRemoteExecutor,
                downloadManager);
      }
    } catch (ExternalDepsException e) {
      throw new SingleExtensionEvalFunctionException(e);
    }
    if (extension == null) {
      return null;
    }

    // Check the lockfile first for that module extension
    LockfileMode lockfileMode = BazelLockFileFunction.LOCKFILE_MODE.get(env);
    Facts lockfileFacts = Facts.EMPTY;
    // Store workspace lockfile facts separately for validation in ERROR mode
    Facts workspaceLockfileFacts = Facts.EMPTY;
    if (!lockfileMode.equals(LockfileMode.OFF)) {
      var lockfiles =
          env.getValuesAndExceptions(
              ImmutableList.of(BazelLockFileValue.KEY, BazelLockFileValue.HIDDEN_KEY));
      BazelLockFileValue workspaceLockfile =
          (BazelLockFileValue) lockfiles.get(BazelLockFileValue.KEY);
      BazelLockFileValue hiddenLockfile =
          (BazelLockFileValue) lockfiles.get(BazelLockFileValue.HIDDEN_KEY);
      if (workspaceLockfile == null || hiddenLockfile == null) {
        return null;
      }
      workspaceLockfileFacts = workspaceLockfile.getFacts().get(extensionId);
      lockfileFacts = workspaceLockfileFacts;
      if (lockfileFacts == null) {
        lockfileFacts = hiddenLockfile.getFacts().getOrDefault(extensionId, Facts.EMPTY);
        workspaceLockfileFacts = Facts.EMPTY;
      }
      var lockedExtensionMap = workspaceLockfile.getModuleExtensions().get(extensionId);
      var lockedExtension =
          lockedExtensionMap == null ? null : lockedExtensionMap.get(extension.getEvalFactors());
      if (lockedExtension == null) {
        lockedExtensionMap = hiddenLockfile.getModuleExtensions().get(extensionId);
        lockedExtension =
            lockedExtensionMap == null ? null : lockedExtensionMap.get(extension.getEvalFactors());
      }
      if (lockedExtension != null) {
        try (SilentCloseable c =
            Profiler.instance()
                .profile(ProfilerTask.BZLMOD, () -> "check lockfile for " + extensionId)) {
          SingleExtensionValue singleExtensionValue =
              tryGettingValueFromLockFile(
                  env,
                  extensionId,
                  extension,
                  usagesValue,
                  extension.getEvalFactors(),
                  lockedExtension,
                  lockfileFacts);
          if (singleExtensionValue != null) {
            return singleExtensionValue;
          }
        } catch (NeedsSkyframeRestartException e) {
          return null;
        }
      }
    }

    // Run that extension!
    RunModuleExtensionResult moduleExtensionResult;
    try {
      moduleExtensionResult =
          extension.run(
              env,
              usagesValue,
              starlarkSemantics,
              extensionId,
              mainRepoMappingValue.repositoryMapping(),
              lockfileFacts);
    } catch (ExternalDepsException e) {
      throw new SingleExtensionEvalFunctionException(e);
    }
    if (moduleExtensionResult == null) {
      return null;
    }
    ImmutableMap<String, RepoSpec> generatedRepoSpecs = moduleExtensionResult.generatedRepoSpecs();
    ModuleExtensionMetadata moduleExtensionMetadata =
        moduleExtensionResult.moduleExtensionMetadata();

    if (!lockfileMode.equals(LockfileMode.OFF)) {
      var nonVisibleRepoNames =
          moduleExtensionResult.recordedInputs().stream()
              .filter(
                  inputAndValue ->
                      inputAndValue.input() instanceof RepoRecordedInput.RecordedRepoMapping
                          && inputAndValue.value() == null)
              .map(entry -> (RepoRecordedInput.RecordedRepoMapping) entry.input())
              .map(RepoRecordedInput.RecordedRepoMapping::apparentName)
              .map(apparentName -> "@" + apparentName)
              .collect(joining(", "));
      if (!nonVisibleRepoNames.isEmpty()) {
        env.getListener()
            .handle(
                Event.warn(
                    String.format(
                        "The module extension %s produced an invalid lockfile entry because it"
                            + " referenced %s. Please report this issue to its maintainers.",
                        extensionId, nonVisibleRepoNames)));
      }
    }
    if (lockfileMode.equals(LockfileMode.ERROR) && !moduleExtensionMetadata.getReproducible()) {
      // The extension is not reproducible and can't be in the lockfile, since an existing (but
      // possibly out-of-date) entry would have been handled by tryGettingValueFromLockFile above.
      throw new SingleExtensionEvalFunctionException(
          ExternalDepsException.withMessage(
              Code.BAD_LOCKFILE,
              "The module extension '%s'%s does not exist in the lockfile",
              extensionId,
              extension.getEvalFactors().isEmpty()
                  ? ""
                  : " for platform " + extension.getEvalFactors()));
    }
    var newFacts = moduleExtensionMetadata.getFacts();
    // In ERROR mode, validate facts only against the workspace lockfile, not the hidden lockfile.
    // The hidden lockfile may contain stale facts from a different version (e.g., after a
    // rollback), which would cause false-positive validation errors.
    if (lockfileMode.equals(LockfileMode.ERROR) && !newFacts.equals(workspaceLockfileFacts)) {
      String reason =
          "the extension '%s' has changed its facts: %s != %s"
              .formatted(
                  extensionId,
                  Starlark.repr(newFacts.value(), starlarkSemantics),
                  Starlark.repr(workspaceLockfileFacts.value(), starlarkSemantics));
      throw createOutdatedLockfileException(reason);
    }

    Optional<LockfileModuleExtensionMetadata> lockfileModuleExtensionMetadata =
        LockfileModuleExtensionMetadata.of(moduleExtensionMetadata);
    Optional<LockFileModuleExtension.WithFactors> lockFileInfo;
    // At this point the extension has been evaluated successfully, but SingleExtensionEvalFunction
    // may still fail if imported repositories were not generated. However, since imports do not
    // influence the evaluation of the extension and the validation also runs when the extension
    // result is taken from the lockfile, we can already populate the lockfile info. This is
    // necessary to prevent the extension from rerunning when only the imports change.
    if (lockfileMode == LockfileMode.UPDATE || lockfileMode == LockfileMode.REFRESH) {
      lockFileInfo =
          Optional.of(
              new LockFileModuleExtension.WithFactors(
                  extension.getEvalFactors(),
                  LockFileModuleExtension.builder()
                      .setBzlTransitiveDigest(extension.getBzlTransitiveDigest())
                      .setUsagesDigest(
                          SingleExtensionUsagesValue.hashForEvaluation(
                              GsonTypeAdapterUtil.SINGLE_EXTENSION_USAGES_VALUE_GSON, usagesValue))
                      .setRecordedInputs(moduleExtensionResult.recordedInputs())
                      .setGeneratedRepoSpecs(generatedRepoSpecs)
                      .setModuleExtensionMetadata(lockfileModuleExtensionMetadata)
                      .build()));
    } else {
      lockFileInfo = Optional.empty();
    }
    return createSingleExtensionValue(
        generatedRepoSpecs,
        lockfileModuleExtensionMetadata,
        extensionId,
        usagesValue,
        lockFileInfo,
        newFacts,
        env);
  }

  /**
   * Tries to get the evaluation result from the lockfile, if it's still up-to-date. Otherwise,
   * returns {@code null}.
   *
   * @throws NeedsSkyframeRestartException in case we need a skyframe restart. Note that we
   *     <em>don't</em> return {@code null} in this case!
   */
  @Nullable
  private SingleExtensionValue tryGettingValueFromLockFile(
      Environment env,
      ModuleExtensionId extensionId,
      RunnableExtension extension,
      SingleExtensionUsagesValue usagesValue,
      ModuleExtensionEvalFactors evalFactors,
      LockFileModuleExtension lockedExtension,
      Facts facts)
      throws SingleExtensionEvalFunctionException,
          InterruptedException,
          NeedsSkyframeRestartException {
    LockfileMode lockfileMode = BazelLockFileFunction.LOCKFILE_MODE.get(env);
    DiffRecorder diffRecorder =
        new DiffRecorder(/* recordMessages= */ lockfileMode.equals(LockfileMode.ERROR));
    try {
      // Put faster diff detections earlier, so that we can short-circuit in UPDATE mode.
      if (!Arrays.equals(
          extension.getBzlTransitiveDigest(), lockedExtension.getBzlTransitiveDigest())) {
        diffRecorder.record(
            "the implementation of the extension '"
                + extensionId
                + "' or one of its transitive .bzl files has changed");
      }
      // Check extension data in lockfile is still valid, disregarding usage information that is not
      // relevant for the evaluation of the extension.
      if (!Arrays.equals(
          SingleExtensionUsagesValue.hashForEvaluation(
              GsonTypeAdapterUtil.SINGLE_EXTENSION_USAGES_VALUE_GSON, usagesValue),
          lockedExtension.getUsagesDigest())) {
        diffRecorder.record("the usages of the extension '" + extensionId + "' have changed");
      }
      Optional<String> reason =
          didRecordedInputsChange(env, directories, lockedExtension.getRecordedInputs());
      if (reason.isPresent()) {
        diffRecorder.record(
            "an input to the extension '" + extensionId + "' changed: " + reason.get());
      }
    } catch (DiffFoundEarlyExitException ignored) {
      // ignored
    }
    // There is intentionally no diff check for facts - they are never invalidated by Bazel.
    if (!diffRecorder.anyDiffsDetected()) {
      return createSingleExtensionValue(
          lockedExtension.getGeneratedRepoSpecs(),
          lockedExtension.getModuleExtensionMetadata(),
          extensionId,
          usagesValue,
          Optional.of(new LockFileModuleExtension.WithFactors(evalFactors, lockedExtension)),
          facts,
          env);
    }
    // Reproducible extensions are always locked in the hidden lockfile to provide best-effort
    // speedups, but should never result in an error if out-of-date.
    if (lockfileMode.equals(LockfileMode.ERROR) && !lockedExtension.isReproducible()) {
      throw createOutdatedLockfileException(diffRecorder.getRecordedDiffMessages());
    }
    return null;
  }

  private static final class DiffFoundEarlyExitException extends Exception {}

  private static final class DiffRecorder {
    private boolean diffDetected = false;
    private final ImmutableList.Builder<String> diffMessages;

    DiffRecorder(boolean recordMessages) {
      diffMessages = recordMessages ? ImmutableList.builder() : null;
    }

    private void record(String message) throws DiffFoundEarlyExitException {
      diffDetected = true;
      if (diffMessages != null) {
        diffMessages.add(message);
      } else {
        throw new DiffFoundEarlyExitException();
      }
    }

    public boolean anyDiffsDetected() {
      return diffDetected;
    }

    public String getRecordedDiffMessages() {
      return String.join(",", diffMessages.build());
    }
  }

  private static Optional<String> didRecordedInputsChange(
      Environment env,
      BlazeDirectories directories,
      List<RepoRecordedInput.WithValue> recordedInputs)
      throws InterruptedException, NeedsSkyframeRestartException {
    // Check inputs in batches to prevent Skyframe cycles caused by outdated dependencies.
    for (ImmutableList<RepoRecordedInput.WithValue> batch :
        RepoRecordedInput.WithValue.splitIntoBatches(recordedInputs)) {
      Optional<String> outdated = RepoRecordedInput.isAnyValueOutdated(env, directories, batch);
      if (env.valuesMissing()) {
        throw new NeedsSkyframeRestartException();
      }
      if (outdated.isPresent()) {
        return outdated;
      }
    }
    return Optional.empty();
  }

  private SingleExtensionValue createSingleExtensionValue(
      ImmutableMap<String, RepoSpec> generatedRepoSpecs,
      Optional<LockfileModuleExtensionMetadata> moduleExtensionMetadata,
      ModuleExtensionId extensionId,
      SingleExtensionUsagesValue usagesValue,
      Optional<LockFileModuleExtension.WithFactors> lockFileInfo,
      Facts facts,
      Environment env)
      throws SingleExtensionEvalFunctionException {
    Optional<RootModuleFileFixup> fixup = Optional.empty();
    if (moduleExtensionMetadata.isPresent()
        && usagesValue.getExtensionUsages().containsKey(ModuleKey.ROOT)) {
      try {
        // TODO: ModuleExtensionMetadata#generateFixup should throw ExternalDepsException instead of
        // EvalException.
        fixup =
            moduleExtensionMetadata
                .get()
                .generateFixup(
                    usagesValue.getExtensionUsages().get(ModuleKey.ROOT),
                    generatedRepoSpecs.keySet());
      } catch (EvalException e) {
        env.getListener().handle(Event.error(e.getInnermostLocation(), e.getMessageWithStack()));
        throw new SingleExtensionEvalFunctionException(
            ExternalDepsException.withMessage(
                Code.BAD_MODULE,
                "error evaluating module extension %s in %s",
                extensionId.extensionName(),
                extensionId.bzlFileLabel()));
      }
    }

    return new SingleExtensionValue(
        generatedRepoSpecs,
        generatedRepoSpecs.keySet().stream()
            .collect(
                toImmutableBiMap(
                    e ->
                        SingleExtensionValue.repositoryName(
                            usagesValue.getExtensionUniqueName(), e),
                    Function.identity())),
        lockFileInfo,
        fixup,
        facts);
  }

  private static SingleExtensionEvalFunctionException createOutdatedLockfileException(
      String reason) {
    return new SingleExtensionEvalFunctionException(
        ExternalDepsException.withMessage(
            Code.BAD_LOCKFILE,
            "MODULE.bazel.lock is no longer up-to-date because %s. Please run `bazel mod deps"
                + " --lockfile_mode=update` to update your lockfile.",
            reason));
  }

  private static final class SingleExtensionEvalFunctionException extends SkyFunctionException {
    SingleExtensionEvalFunctionException(ExternalDepsException cause) {
      super(cause, Transience.PERSISTENT);
    }
  }
}
