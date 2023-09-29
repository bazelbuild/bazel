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
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.collect.Maps.transformValues;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableTable;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.LockfileMode;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
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
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.function.Function;
import java.util.function.Supplier;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
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
    ImmutableMap<String, String> extensionEnvVars =
        RepositoryFunction.getEnvVarValues(env, ImmutableSet.copyOf(extension.getEnvVariables()));
    if (extensionEnvVars == null) {
      return null;
    }
    byte[] bzlTransitiveDigest =
        BazelModuleContext.of(bzlLoadValue.getModule()).bzlTransitiveDigest();

    // Check the lockfile first for that module extension
    LockfileMode lockfileMode = BazelLockFileFunction.LOCKFILE_MODE.get(env);
    ModuleExtensionEvalFactors extensionFactors =
        ModuleExtensionEvalFactors.create(
            extension.getOsDependent() ? OS.getCurrent().toString() : "",
            extension.getArchDependent() ? OS_ARCH.value() : "");
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
                extensionFactors,
                extensionEnvVars,
                usagesValue,
                bzlTransitiveDigest,
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
        runModuleExtension(
            extensionId, extension, usagesValue, bzlLoadValue.getModule(), starlarkSemantics, env);
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
                  extensionFactors,
                  LockFileModuleExtension.builder()
                      .setBzlTransitiveDigest(bzlTransitiveDigest)
                      .setAccumulatedFileDigests(moduleExtensionResult.getAccumulatedFileDigests())
                      .setEnvVariables(extensionEnvVars)
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
                "The module extension '%s''%s' does not exist in the lockfile",
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
    var trimmedLockedUsages =
        ImmutableMap.copyOf(
            transformValues(lockedExtensionUsages, ModuleExtensionUsage::trimForEvaluation));
    var trimmedUsages =
        ImmutableMap.copyOf(
            transformValues(
                usagesValue.getExtensionUsages(), ModuleExtensionUsage::trimForEvaluation));
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

  @Nullable
  private RunModuleExtensionResult runModuleExtension(
      ModuleExtensionId extensionId,
      ModuleExtension extension,
      SingleExtensionUsagesValue usagesValue,
      Module module,
      StarlarkSemantics starlarkSemantics,
      Environment env)
      throws SingleExtensionEvalFunctionException, InterruptedException {
    ModuleExtensionEvalStarlarkThreadContext threadContext =
        new ModuleExtensionEvalStarlarkThreadContext(
            usagesValue.getExtensionUniqueName() + "~",
            extensionId.getBzlFileLabel().getPackageIdentifier(),
            BazelModuleContext.of(module).repoMapping(),
            directories,
            env.getListener());
    ModuleExtensionContext moduleContext;
    Optional<ModuleExtensionMetadata> moduleExtensionMetadata;
    try (Mutability mu =
        Mutability.create("module extension", usagesValue.getExtensionUniqueName())) {
      StarlarkThread thread = new StarlarkThread(mu, starlarkSemantics);
      thread.setPrintHandler(Event.makeDebugPrintHandler(env.getListener()));
      moduleContext = createContext(env, usagesValue, starlarkSemantics, extensionId, extension);
      threadContext.storeInThread(thread);
      try (SilentCloseable c =
          Profiler.instance()
              .profile(
                  ProfilerTask.BZLMOD,
                  () -> "evaluate module extension: " + extensionId.asTargetString())) {
        Object returnValue =
            Starlark.fastcall(
                thread, extension.getImplementation(), new Object[] {moduleContext}, new Object[0]);
        if (returnValue != Starlark.NONE && !(returnValue instanceof ModuleExtensionMetadata)) {
          throw new SingleExtensionEvalFunctionException(
              ExternalDepsException.withMessage(
                  ExternalDeps.Code.BAD_MODULE,
                  "expected module extension %s in %s to return None or extension_metadata, got %s",
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
      ModuleExtensionId extensionId,
      ModuleExtension extension)
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
