// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.collect.nestedset.Order.STABLE_ORDER;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactResolver;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.PackageRootResolutionException;
import com.google.devtools.build.lib.actions.PackageRootResolver;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.Platform;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction.DotdFile;
import com.google.devtools.build.lib.rules.cpp.HeaderDiscovery;
import com.google.devtools.build.lib.rules.cpp.HeaderDiscovery.DotdPruningMode;
import com.google.devtools.build.lib.rules.cpp.IncludeScanningContext;
import com.google.devtools.build.lib.util.DependencySet;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.concurrent.GuardedBy;

/**
 * An action that compiles objc or objc++ source.
 *
 * <p>We don't use a plain SpawnAction here because we implement .d input pruning, which requires
 * post-execution filtering of input artifacts.
 *
 * <p>We don't use a CppCompileAction because the ObjcCompileAction uses custom logic instead of the
 * CROSSTOOL to construct its command line.
 */
public class ObjcCompileAction extends SpawnAction {

  private final DotdFile dotdFile;
  private final Artifact sourceFile;
  private final NestedSet<Artifact> mandatoryInputs;
  private final HeaderDiscovery.DotdPruningMode dotdPruningPlan;

  // This can be read/written from multiple threads, so accesses must be synchronized.
  @GuardedBy("this")
  private boolean inputsKnown = false;
  
  private static final String GUID = "a00d5bac-a72c-4f0f-99a7-d5fdc6072137";

  private ObjcCompileAction(
      ActionOwner owner,
      Iterable<Artifact> tools,
      Iterable<Artifact> inputs,
      Iterable<Artifact> outputs,
      ResourceSet resourceSet,
      CommandLine argv,
      ImmutableMap<String, String> environment,
      ImmutableMap<String, String> executionInfo,
      String progressMessage,
      ImmutableMap<PathFragment, Artifact> inputManifests,
      String mnemonic,
      boolean executeUnconditionally,
      ExtraActionInfoSupplier<?> extraActionInfoSupplier,
      DotdFile dotdFile,
      Artifact sourceFile,
      NestedSet<Artifact> mandatoryInputs,
      HeaderDiscovery.DotdPruningMode dotdPruningPlan) {
    super(
        owner,
        tools,
        inputs,
        outputs,
        resourceSet,
        argv,
        environment,
        ImmutableSet.<String>of(),
        executionInfo,
        progressMessage,
        inputManifests,
        mnemonic,
        executeUnconditionally,
        extraActionInfoSupplier);

    this.dotdFile = dotdFile;
    this.sourceFile = sourceFile;
    this.mandatoryInputs = mandatoryInputs;
    this.dotdPruningPlan = dotdPruningPlan;
    this.inputsKnown = (dotdPruningPlan == DotdPruningMode.DO_NOT_USE);
  }

  /** Returns the DotdPruningPlan for this compile */
  @VisibleForTesting
  public HeaderDiscovery.DotdPruningMode getDotdPruningPlan() {
    return dotdPruningPlan;
  }

  @Override
  public synchronized boolean inputsKnown() {
    return inputsKnown;
  }
  
  @Override
  public boolean discoversInputs() {
    return true;
  }

  @Override
  public Iterable<Artifact> discoverInputs(ActionExecutionContext actionExecutionContext) {
    // We do not use include scanning for objc
    return null;
  }
 
  // Keep in sync with {@link CppCompileAction#resolveInputsFromCache}
  @Override
  public Iterable<Artifact> resolveInputsFromCache(
      ArtifactResolver artifactResolver,
      PackageRootResolver resolver,
      Collection<PathFragment> inputPaths)
      throws PackageRootResolutionException, InterruptedException {
    // Note that this method may trigger a violation of the desirable invariant that getInputs()
    // is a superset of getMandatoryInputs(). See bug about an "action not in canonical form"
    // error message and the integration test test_crosstool_change_and_failure().
    Map<PathFragment, Artifact> allowedDerivedInputsMap = getAllowedDerivedInputsMap();
    List<Artifact> inputs = new ArrayList<>();
    List<PathFragment> unresolvedPaths = new ArrayList<>();
    for (PathFragment execPath : inputPaths) {
      Artifact artifact = allowedDerivedInputsMap.get(execPath);
      if (artifact != null) {
        inputs.add(artifact);
      } else {
        // Remember this execPath, we will try to resolve it as a source artifact.
        unresolvedPaths.add(execPath);
      }
    }

    Map<PathFragment, Artifact> resolvedArtifacts =
        artifactResolver.resolveSourceArtifacts(unresolvedPaths, resolver);
    if (resolvedArtifacts == null) {
      // We are missing some dependencies. We need to rerun this update later.
      return null;
    }

    for (PathFragment execPath : unresolvedPaths) {
      Artifact artifact = resolvedArtifacts.get(execPath);
      // If PathFragment cannot be resolved into the artifact - ignore it. This could happen if
      // rule definition has changed and action no longer depends on, e.g., additional source file
      // in the separate package and that package is no longer referenced anywhere else.
      // It is safe to ignore such paths because dependency checker would identify change in inputs
      // (ignored path was used before) and will force action execution.
      if (artifact != null) {
        inputs.add(artifact);
      }
    }
    return inputs;    
  }
  
  @Override
  public synchronized void updateInputs(Iterable<Artifact> inputs) {
    inputsKnown = true;
    synchronized (this) {
      setInputs(inputs);
    }
  }  
  
  @Override
  public ImmutableSet<Artifact> getMandatoryOutputs() {
    return ImmutableSet.of(dotdFile.artifact());
  }

  @Override
  public void execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    super.execute(actionExecutionContext);

    if (dotdPruningPlan == HeaderDiscovery.DotdPruningMode.USE) {
      Executor executor = actionExecutionContext.getExecutor();
      IncludeScanningContext scanningContext = executor.getContext(IncludeScanningContext.class);
      NestedSet<Artifact> discoveredInputs =
          discoverInputsFromDotdFiles(
              executor.getExecRoot(), scanningContext.getArtifactResolver());

      updateActionInputs(discoveredInputs);
    }
  }

  @VisibleForTesting
  public NestedSet<Artifact> discoverInputsFromDotdFiles(
      Path execRoot, ArtifactResolver artifactResolver) throws ActionExecutionException {
    if (dotdFile == null) {
      return NestedSetBuilder.<Artifact>stableOrder().build();
    }
    return new HeaderDiscovery.Builder()
        .setAction(this)
        .setSourceFile(sourceFile)
        .setDotdFile(dotdFile)
        .setDependencySet(processDepset(execRoot))
        .setPermittedSystemIncludePrefixes(ImmutableList.<Path>of())
        .setAllowedDerivedinputsMap(getAllowedDerivedInputsMap())
        .build()
        .discoverInputsFromDotdFiles(execRoot, artifactResolver);
  }

  private DependencySet processDepset(Path execRoot) throws ActionExecutionException {
    try {
      DependencySet depSet = new DependencySet(execRoot);
      return depSet.read(dotdFile.getPath());
    } catch (IOException e) {
      // Some kind of IO or parse exception--wrap & rethrow it to stop the build.
      throw new ActionExecutionException("error while parsing .d file", e, this, false);
    }
  }

  /** Utility function that adds artifacts to an input map, but only if they are sources. */
  private void addToMapIfSource(Map<PathFragment, Artifact> map, Iterable<Artifact> artifacts) {
    for (Artifact artifact : artifacts) {
      if (!artifact.isSourceArtifact()) {
        map.put(artifact.getExecPath(), artifact);
      }
    }
  }

  private Map<PathFragment, Artifact> getAllowedDerivedInputsMap() {
    Map<PathFragment, Artifact> allowedDerivedInputMap = new HashMap<>();
    addToMapIfSource(allowedDerivedInputMap, getInputs());
    allowedDerivedInputMap.put(sourceFile.getExecPath(), sourceFile);
    return allowedDerivedInputMap;
  }

  /**
   * Recalculates this action's live input collection, including sources, middlemen.
   *
   * @throws ActionExecutionException iff any errors happen during update.
   */
  @VisibleForTesting
  @ThreadCompatible
  public final synchronized void updateActionInputs(NestedSet<Artifact> discoveredInputs)
      throws ActionExecutionException {
    inputsKnown = false;
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.stableOrder();
    Profiler.instance().startTask(ProfilerTask.ACTION_UPDATE, this);
    try {
      inputs.addTransitive(mandatoryInputs);
      inputs.addTransitive(discoveredInputs);
      inputsKnown = true;
    } finally {
      Profiler.instance().completeTask(ProfilerTask.ACTION_UPDATE);
      setInputs(inputs.build());
    }
  }

  @Override
  public String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addString(GUID);
    f.addString(super.computeKey());
    f.addBoolean(dotdFile.artifact() == null);
    f.addBoolean(dotdPruningPlan == HeaderDiscovery.DotdPruningMode.USE);
    f.addPath(dotdFile.getSafeExecPath());
    return f.hexDigestAndReset();
  }

  /** A Builder for ObjcCompileAction */
  public static class Builder extends SpawnAction.Builder {

    private DotdFile dotdFile;
    private Artifact sourceFile;
    private final NestedSetBuilder<Artifact> mandatoryInputs = new NestedSetBuilder<>(STABLE_ORDER);
    private HeaderDiscovery.DotdPruningMode dotdPruningPlan;

    /**
     * Creates a new compile action builder with apple environment variables set that are typically
     * needed by the apple toolchain.
     */
    public static ObjcCompileAction.Builder createObjcCompileActionBuilderWithAppleEnv(
        AppleConfiguration appleConfiguration, Platform targetPlatform) {
      return (Builder)
          new ObjcCompileAction.Builder()
              .setExecutionInfo(ObjcRuleClasses.darwinActionExecutionRequirement())
              .setEnvironment(
                  ObjcRuleClasses.appleToolchainEnvironment(appleConfiguration, targetPlatform));
    }

    @Override
    public Builder addTools(Iterable<Artifact> artifacts) {
      super.addTools(artifacts);
      mandatoryInputs.addAll(artifacts);
      return this;
    }

    /** Sets a .d file that will used to prune input headers */
    public Builder setDotdFile(DotdFile dotdFile) {
      Preconditions.checkNotNull(dotdFile);
      this.dotdFile = dotdFile;
      return this;
    }

    /** Sets the source file that is being compiled in this action */
    public Builder setSourceFile(Artifact sourceFile) {
      Preconditions.checkNotNull(sourceFile);
      this.sourceFile = sourceFile;
      this.mandatoryInputs.add(sourceFile);
      this.addInput(sourceFile);
      return this;
    }

    /** Add an input that cannot be pruned */
    public Builder addMandatoryInput(Artifact input) {
      Preconditions.checkNotNull(input);
      this.mandatoryInputs.add(input);
      this.addInput(input);
      return this;
    }

    /** Add inputs that cannot be pruned */
    public Builder addMandatoryInputs(Iterable<Artifact> input) {
      Preconditions.checkNotNull(input);
      this.mandatoryInputs.addAll(input);
      this.addInputs(input);
      return this;
    }

    /** Add inputs that cannot be pruned */
    public Builder addTransitiveMandatoryInputs(NestedSet<Artifact> input) {
      Preconditions.checkNotNull(input);
      this.mandatoryInputs.addTransitive(input);
      this.addTransitiveInputs(input);
      return this;
    }

    /** Indicates that this compile action should perform .d pruning */
    public Builder setDotdPruningPlan(HeaderDiscovery.DotdPruningMode dotdPruningPlan) {
      Preconditions.checkNotNull(dotdPruningPlan);
      this.dotdPruningPlan = dotdPruningPlan;
      return this;
    }

    @Override
    protected SpawnAction createSpawnAction(
        ActionOwner owner,
        NestedSet<Artifact> tools,
        NestedSet<Artifact> inputsAndTools,
        ImmutableList<Artifact> outputs,
        ResourceSet resourceSet,
        CommandLine actualCommandLine,
        ImmutableMap<String, String> env,
        ImmutableSet<String> clientEnvironmentVariables,
        ImmutableMap<String, String> executionInfo,
        String progressMessage,
        ImmutableMap<PathFragment, Artifact> inputAndToolManifests,
        String mnemonic) {
      return new ObjcCompileAction(
          owner,
          tools,
          inputsAndTools,
          outputs,
          resourceSet,
          actualCommandLine,
          env,
          executionInfo,
          progressMessage,
          inputAndToolManifests,
          mnemonic,
          executeUnconditionally,
          extraActionInfoSupplier,
          dotdFile,
          sourceFile,
          mandatoryInputs.build(),
          dotdPruningPlan);
    }
  }
}
