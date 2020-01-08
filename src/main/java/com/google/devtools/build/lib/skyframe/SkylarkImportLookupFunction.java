// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;


import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.base.Throwables;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.InconsistentFilesystemException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.SkylarkExportable;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.skyframe.SkylarkImportLookupValue.SkylarkImportLookupKey;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.LoadStatement;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.StarlarkFile;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.StarlarkThread.Extension;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.RecordingSkyFunctionEnvironment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/**
 * A Skyframe function to look up and import a single Skylark extension.
 *
 * <p>Given a {@link Label} referencing a Skylark file, attempts to locate the file and load it. The
 * Label must be absolute, and must not reference the special {@code external} package. If loading
 * is successful, returns a {@link SkylarkImportLookupValue} that encapsulates the loaded {@link
 * Extension} and {@link SkylarkFileDependency} information. If loading is unsuccessful, throws a
 * {@link SkylarkImportLookupFunctionException} that encapsulates the cause of the failure.
 */
public class SkylarkImportLookupFunction implements SkyFunction {

  private final RuleClassProvider ruleClassProvider;
  private final PackageFactory packageFactory;
  private final int starlarkImportLookupValueCacheSize;
  private Cache<SkyKey, CachedSkylarkImportLookupValueAndDeps> skylarkImportLookupValueCache;
  private CachedSkylarkImportLookupValueAndDepsBuilderFactory
      cachedSkylarkImportLookupValueAndDepsBuilderFactory =
          new CachedSkylarkImportLookupValueAndDepsBuilderFactory();

  private static final Logger logger =
      Logger.getLogger(SkylarkImportLookupFunction.class.getName());

  public SkylarkImportLookupFunction(
      RuleClassProvider ruleClassProvider, PackageFactory packageFactory) {
    this(ruleClassProvider, packageFactory, /*starlarkImportLookupValueCacheSize=*/ -1);
  }

  public SkylarkImportLookupFunction(
      RuleClassProvider ruleClassProvider,
      PackageFactory packageFactory,
      int starlarkImportLookupValueCacheSize) {
    this.ruleClassProvider = ruleClassProvider;
    this.packageFactory = packageFactory;
    this.starlarkImportLookupValueCacheSize = starlarkImportLookupValueCacheSize;
  }

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    SkylarkImportLookupKey key = (SkylarkImportLookupKey) skyKey.argument();
    try {
      return computeInternal(
          key.importLabel,
          key.inWorkspace,
          key.workspaceChunk,
          key.workspacePath,
          env,
          /*visitedNested=*/ null,
          /*inlineCachedValueBuilder=*/ null,
          /*visitedDepsInToplevelLoad=*/ null);
    } catch (InconsistentFilesystemException e) {
      throw new SkylarkImportLookupFunctionException(e, Transience.PERSISTENT);
    } catch (SkylarkImportFailedException e) {
      throw new SkylarkImportLookupFunctionException(e);
    }
  }

  @Nullable
  SkylarkImportLookupValue computeWithInlineCalls(
      SkyKey skyKey,
      Environment env,
      Map<SkylarkImportLookupKey, CachedSkylarkImportLookupValueAndDeps> visitedDepsInToplevelLoad)
      throws InconsistentFilesystemException, SkylarkImportFailedException, InterruptedException {
    // We use the visitedNested set to track if there are any cyclic dependencies when loading the
    // skylark file and the visitedDepsInToplevelLoad set to avoid re-registering previously seen
    // dependencies. Note that the visitedNested set must use insertion order to display the correct
    // error.
    CachedSkylarkImportLookupValueAndDeps cachedSkylarkImportLookupValueAndDeps =
        computeWithInlineCallsInternal(
            skyKey,
            env,
            /*visitedNested=*/ new LinkedHashSet<>(),
            /*visitedDepsInToplevelLoad=*/ visitedDepsInToplevelLoad);
    if (cachedSkylarkImportLookupValueAndDeps == null) {
      return null;
    }
    return cachedSkylarkImportLookupValueAndDeps.getValue();
  }

  @Nullable
  private CachedSkylarkImportLookupValueAndDeps computeWithInlineCallsInternal(
      SkyKey skyKey,
      Environment env,
      Set<Label> visitedNested,
      Map<SkylarkImportLookupKey, CachedSkylarkImportLookupValueAndDeps> visitedDepsInToplevelLoad)
      throws InconsistentFilesystemException, SkylarkImportFailedException, InterruptedException {
    SkylarkImportLookupKey key = (SkylarkImportLookupKey) skyKey.argument();
    Label importLabel = key.importLabel;

    // If we've visited a SkylarkImportLookupValue through some other load path for a given package,
    // we must use the existing value to preserve reference equality between Starlark values that
    // ought to be the same. See b/138598337 for details.
    CachedSkylarkImportLookupValueAndDeps cachedSkylarkImportLookupValueAndDeps =
        visitedDepsInToplevelLoad.get(key);
    if (cachedSkylarkImportLookupValueAndDeps == null) {
      // Note that we can't block other threads on the computation of this value due to a potential
      // deadlock on a cycle. Although we are repeating some work, it is possible we have an import
      // cycle where one thread starts at one side of the cycle and the other thread starts at the
      // other side, and they then wait forever on the results of each others computations.
      cachedSkylarkImportLookupValueAndDeps = skylarkImportLookupValueCache.getIfPresent(skyKey);
      if (cachedSkylarkImportLookupValueAndDeps != null) {
        cachedSkylarkImportLookupValueAndDeps.traverse(
            env::registerDependencies, visitedDepsInToplevelLoad);
      }
    }
    if (cachedSkylarkImportLookupValueAndDeps != null) {
      return cachedSkylarkImportLookupValueAndDeps;
    }

    if (!visitedNested.add(importLabel)) {
      ImmutableList<Label> cycle =
          CycleUtils.splitIntoPathAndChain(Predicates.equalTo(importLabel), visitedNested).second;
      throw new SkylarkImportFailedException("Starlark import cycle: " + cycle);
    }

    CachedSkylarkImportLookupValueAndDeps.Builder inlineCachedValueBuilder =
        cachedSkylarkImportLookupValueAndDepsBuilderFactory
            .newCachedSkylarkImportLookupValueAndDepsBuilder();
    Preconditions.checkState(
        !(env instanceof RecordingSkyFunctionEnvironment),
        "Found nested RecordingSkyFunctionEnvironment but it should have been stripped: %s",
        env);
    RecordingSkyFunctionEnvironment recordingEnv =
        new RecordingSkyFunctionEnvironment(
            env,
            inlineCachedValueBuilder::addDep,
            inlineCachedValueBuilder::addDeps,
            inlineCachedValueBuilder::noteException);
    SkylarkImportLookupValue value =
        computeInternal(
            importLabel,
            key.inWorkspace,
            key.workspaceChunk,
            key.workspacePath,
            recordingEnv,
            Preconditions.checkNotNull(visitedNested, importLabel),
            inlineCachedValueBuilder,
            visitedDepsInToplevelLoad);
    // All imports traversed, this key can no longer be part of a cycle.
    Preconditions.checkState(visitedNested.remove(importLabel), importLabel);

    if (value != null) {
      inlineCachedValueBuilder.setValue(value);
      inlineCachedValueBuilder.setKey(key);
      cachedSkylarkImportLookupValueAndDeps = inlineCachedValueBuilder.build();
      visitedDepsInToplevelLoad.put(key, cachedSkylarkImportLookupValueAndDeps);
      skylarkImportLookupValueCache.put(skyKey, cachedSkylarkImportLookupValueAndDeps);
    }
    return cachedSkylarkImportLookupValueAndDeps;
  }

  public void resetCache() {
    if (skylarkImportLookupValueCache != null) {
      logger.info(
          "Starlark inlining cache stats from earlier build: "
              + skylarkImportLookupValueCache.stats());
    }
    cachedSkylarkImportLookupValueAndDepsBuilderFactory =
        new CachedSkylarkImportLookupValueAndDepsBuilderFactory();
    Preconditions.checkState(
        starlarkImportLookupValueCacheSize >= 0,
        "Expected positive skylark cache size if caching. %s",
        starlarkImportLookupValueCacheSize);
    skylarkImportLookupValueCache =
        CacheBuilder.newBuilder()
            .concurrencyLevel(BlazeInterners.concurrencyLevel())
            .maximumSize(starlarkImportLookupValueCacheSize)
            .recordStats()
            .build();
  }

  private static ContainingPackageLookupValue getContainingPackageLookupValue(
      Environment env, Label fileLabel)
      throws InconsistentFilesystemException, SkylarkImportFailedException, InterruptedException {
    PathFragment dir = Label.getContainingDirectory(fileLabel);
    PackageIdentifier dirId =
        PackageIdentifier.create(fileLabel.getPackageIdentifier().getRepository(), dir);
    ContainingPackageLookupValue containingPackageLookupValue;
    try {
      containingPackageLookupValue =
          (ContainingPackageLookupValue)
              env.getValueOrThrow(
                  ContainingPackageLookupValue.key(dirId),
                  BuildFileNotFoundException.class,
                  InconsistentFilesystemException.class);
    } catch (BuildFileNotFoundException e) {
      throw SkylarkImportFailedException.errorReadingFile(
          fileLabel.toPathFragment(), new ErrorReadingSkylarkExtensionException(e));
    }
    if (containingPackageLookupValue == null) {
      return null;
    }
    // Ensure the label doesn't cross package boundaries.
    if (!containingPackageLookupValue.hasContainingPackage()) {
      throw SkylarkImportFailedException.noBuildFile(
          fileLabel, containingPackageLookupValue.getReasonForNoContainingPackage());
    }
    if (!containingPackageLookupValue
        .getContainingPackageName()
        .equals(fileLabel.getPackageIdentifier())) {
      throw SkylarkImportFailedException.labelCrossesPackageBoundary(
          fileLabel, containingPackageLookupValue);
    }
    return containingPackageLookupValue;
  }

  // It is vital that we don't return any value if any call to env#getValue(s)OrThrow throws an
  // exception. We are allowed to wrap the thrown exception and rethrow it for any calling functions
  // to handle though.
  @Nullable
  private SkylarkImportLookupValue computeInternal(
      Label fileLabel,
      boolean inWorkspace,
      int workspaceChunk,
      RootedPath workspacePath,
      Environment env,
      @Nullable Set<Label> visitedNested,
      @Nullable CachedSkylarkImportLookupValueAndDeps.Builder inlineCachedValueBuilder,
      @Nullable
          Map<SkylarkImportLookupKey, CachedSkylarkImportLookupValueAndDeps>
              visitedDepsInToplevelLoad)
      throws InconsistentFilesystemException, SkylarkImportFailedException, InterruptedException {
    PathFragment filePath = fileLabel.toPathFragment();

    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }

    if (getContainingPackageLookupValue(env, fileLabel) == null) {
      return null;
    }

    // Load the AST corresponding to this file.
    ASTFileLookupValue astLookupValue;
    try {
      SkyKey astLookupKey = ASTFileLookupValue.key(fileLabel);
      astLookupValue = (ASTFileLookupValue) env.getValueOrThrow(astLookupKey,
          ErrorReadingSkylarkExtensionException.class, InconsistentFilesystemException.class);
    } catch (ErrorReadingSkylarkExtensionException e) {
      throw SkylarkImportFailedException.errorReadingFile(filePath, e);
    }
    if (astLookupValue == null) {
      return null;
    }
    if (!astLookupValue.lookupSuccessful()) {
      // Skylark import files have to exist.
      throw new SkylarkImportFailedException(astLookupValue.getErrorMsg());
    }
    StarlarkFile file = astLookupValue.getAST();
    if (!file.ok()) {
      throw SkylarkImportFailedException.skylarkErrors(filePath);
    }

    // Process the load statements in the file,
    // resolving labels relative to the current repo mapping.
    ImmutableMap<RepositoryName, RepositoryName> repoMapping =
        getRepositoryMapping(workspaceChunk, workspacePath, fileLabel, env);
    if (repoMapping == null) {
      return null;
    }
    Map<String, Label> loadMap =
        getLoadMap(env.getListener(), file, fileLabel.getPackageIdentifier(), repoMapping);
    if (loadMap == null) {
      // malformed load statements
      throw SkylarkImportFailedException.skylarkErrors(filePath);
    }

    // Look up and load the imports.
    List<SkyKey> importLookupKeys = Lists.newArrayListWithExpectedSize(loadMap.size());
    for (Label importLabel : loadMap.values()) {
      if (inWorkspace) {
        importLookupKeys.add(
            SkylarkImportLookupValue.keyInWorkspace(importLabel, workspaceChunk, workspacePath));
      } else {
        importLookupKeys.add(SkylarkImportLookupValue.key(importLabel));
      }
    }
    Map<SkyKey, SkyValue> skylarkImportMap =
        (visitedNested == null)
            ? computeSkylarkImportMapNoInlining(env, importLookupKeys, file.getStartLocation())
            : computeSkylarkImportMapWithInlining(
                env,
                importLookupKeys,
                fileLabel,
                visitedNested,
                inlineCachedValueBuilder,
                visitedDepsInToplevelLoad);
    // skylarkImportMap is null when skyframe deps are unavailable.
    if (skylarkImportMap == null) {
      return null;
    }

    // Process the loaded imports.
    Map<String, Extension> extensionsForImports = Maps.newHashMapWithExpectedSize(loadMap.size());
    ImmutableList.Builder<SkylarkFileDependency> fileDependencies =
        ImmutableList.builderWithExpectedSize(loadMap.size());
    for (Map.Entry<String, Label> importEntry : loadMap.entrySet()) {
      String importString = importEntry.getKey();
      Label importLabel = importEntry.getValue();
      SkyKey keyForLabel;
      if (inWorkspace) {
        keyForLabel =
            SkylarkImportLookupValue.keyInWorkspace(importLabel, workspaceChunk, workspacePath);
      } else {
        keyForLabel = SkylarkImportLookupValue.key(importLabel);
      }
      SkylarkImportLookupValue importLookupValue =
          (SkylarkImportLookupValue) skylarkImportMap.get(keyForLabel);
      extensionsForImports.put(importString, importLookupValue.getEnvironmentExtension());
      fileDependencies.add(importLookupValue.getDependency());
    }

    // #createExtension does not request values from the Environment. It may post events to the
    // Environment, but events do not matter when caching SkylarkImportLookupValues.
    Extension extension =
        createExtension(
            file,
            fileLabel,
            extensionsForImports,
            starlarkSemantics,
            env,
            inWorkspace,
            repoMapping);
    SkylarkImportLookupValue result =
        new SkylarkImportLookupValue(
            extension, new SkylarkFileDependency(fileLabel, fileDependencies.build()));
    return result;
  }

  private static ImmutableMap<RepositoryName, RepositoryName> getRepositoryMapping(
      int workspaceChunk, RootedPath workspacePath, Label enclosingFileLabel, Environment env)
      throws InterruptedException {

    // There is no previous workspace chunk
    if (workspaceChunk == 0) {
      return ImmutableMap.of();
    }

    ImmutableMap<RepositoryName, RepositoryName> repositoryMapping;
    // We are fully done with workspace evaluation so we should get the mappings from the
    // final RepositoryMappingValue
    if (workspaceChunk == -1) {
      PackageIdentifier packageIdentifier = enclosingFileLabel.getPackageIdentifier();
      RepositoryMappingValue repositoryMappingValue =
          (RepositoryMappingValue)
              env.getValue(RepositoryMappingValue.key(packageIdentifier.getRepository()));
      if (repositoryMappingValue == null) {
        return null;
      }
      repositoryMapping = repositoryMappingValue.getRepositoryMapping();
    } else { // Still during workspace file evaluation
      SkyKey workspaceFileKey = WorkspaceFileValue.key(workspacePath, workspaceChunk - 1);
      WorkspaceFileValue workspaceFileValue = (WorkspaceFileValue) env.getValue(workspaceFileKey);
      // Note: we know for sure that the requested WorkspaceFileValue is fully computed so we do not
      // need to check if it is null
      repositoryMapping =
          workspaceFileValue
              .getRepositoryMapping()
              .getOrDefault(
                  enclosingFileLabel.getPackageIdentifier().getRepository(), ImmutableMap.of());
    }
    return repositoryMapping;
  }

  /**
   * Returns a mapping from each load string in the BUILD or .bzl file to the Label it resolves to.
   * Labels are resolved relative to {@code base}, the file's package. If any load statement is
   * malformed, getLoadMap reports one or more errors to the handler and returns null.
   */
  @Nullable
  static Map<String, Label> getLoadMap(
      EventHandler handler,
      StarlarkFile file,
      PackageIdentifier base,
      ImmutableMap<RepositoryName, RepositoryName> repoMapping) {
    Preconditions.checkArgument(!base.getRepository().isDefault());

    // It's redundant that getRelativeWithRemapping needs a Label;
    // a PackageIdentifier should suffice. Make one here.
    Label buildLabel = getBUILDLabel(base);

    boolean ok = true;
    Map<String, Label> loadMap = Maps.newHashMap();
    for (Statement stmt : file.getStatements()) {
      if (stmt instanceof LoadStatement) {
        LoadStatement load = (LoadStatement) stmt;
        String module = load.getImport().getValue();

        // Parse the load statement's module string as a label.
        // It must end in .bzl and not be in package "//external".
        try {
          Label label = buildLabel.getRelativeWithRemapping(module, repoMapping);
          if (!label.getName().endsWith(".bzl")) {
            throw new LabelSyntaxException("The label must reference a file with extension '.bzl'");
          }
          if (label.getPackageIdentifier().equals(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER)) {
            throw new LabelSyntaxException(
                "Starlark files may not be loaded from the //external package");
          }
          loadMap.put(module, label);
        } catch (LabelSyntaxException ex) {
          handler.handle(
              Event.error(
                  load.getImport().getStartLocation(), "in load statement: " + ex.getMessage()));
          ok = false;
        }
      }
    }
    return ok ? loadMap : null;
  }

  private static Label getBUILDLabel(PackageIdentifier pkgid) {
    try {
      return Label.create(pkgid, "BUILD");
    } catch (LabelSyntaxException e) {
      // Shouldn't happen; the Label is well-formed by construction.
      throw new IllegalStateException(e);
    }
  }

  /**
   * Compute the SkylarkImportLookupValue for all given SkyKeys using vanilla skyframe evaluation,
   * returning {@code null} if skyframe deps were missing and have been requested.
   */
  @Nullable
  private static Map<SkyKey, SkyValue> computeSkylarkImportMapNoInlining(
      Environment env, List<SkyKey> importLookupKeys, Location locationForErrors)
      throws SkylarkImportFailedException, InterruptedException {
    Map<SkyKey, SkyValue> skylarkImportMap =
        Maps.newHashMapWithExpectedSize(importLookupKeys.size());
    Map<SkyKey, ValueOrException<SkylarkImportFailedException>> values =
        env.getValuesOrThrow(importLookupKeys, SkylarkImportFailedException.class);
    // NOTE: Iterating over imports in the order listed in the file.
    for (SkyKey key : importLookupKeys) {
      try {
        skylarkImportMap.put(key, values.get(key).get());
      } catch (SkylarkImportFailedException exn) {
        throw new SkylarkImportFailedException(
            "in " + locationForErrors.file() + ": " + exn.getMessage());
      }
    }
    return env.valuesMissing() ? null : skylarkImportMap;
  }

  /**
   * Compute the SkylarkImportLookupValue for all given SkyKeys by reusing this instance of the
   * SkylarkImportLookupFunction, bypassing traditional skyframe evaluation, returning {@code null}
   * if skyframe deps were missing and have been requested.
   */
  @Nullable
  private Map<SkyKey, SkyValue> computeSkylarkImportMapWithInlining(
      Environment env,
      List<SkyKey> importLookupKeys,
      Label fileLabel,
      Set<Label> visitedNested,
      CachedSkylarkImportLookupValueAndDeps.Builder inlineCachedValueBuilder,
      Map<SkylarkImportLookupKey, CachedSkylarkImportLookupValueAndDeps> visitedDepsInToplevelLoad)
      throws InterruptedException, SkylarkImportFailedException, InconsistentFilesystemException {
    Preconditions.checkNotNull(
        inlineCachedValueBuilder,
        "Expected inline cached value builder to be not-null when inlining.");
    Preconditions.checkState(
        env instanceof RecordingSkyFunctionEnvironment,
        "Expected to be recording dep requests when inlining SkylarkImportLookupFunction: %s",
        fileLabel);
    Environment strippedEnv = ((RecordingSkyFunctionEnvironment) env).getDelegate();
    Map<SkyKey, SkyValue> skylarkImportMap =
        Maps.newHashMapWithExpectedSize(importLookupKeys.size());
    Exception deferredException = null;
    boolean valuesMissing = false;
    // NOTE: Iterating over imports in the order listed in the file.
    for (SkyKey importLookupKey : importLookupKeys) {
      CachedSkylarkImportLookupValueAndDeps cachedValue;
      try {
        cachedValue =
            computeWithInlineCallsInternal(
                importLookupKey, strippedEnv, visitedNested, visitedDepsInToplevelLoad);
      } catch (SkylarkImportFailedException | InconsistentFilesystemException e) {
        // For determinism's sake while inlining, preserve the first exception and continue to run
        // subsequently listed imports to completion/exception, loading all transitive deps anyway.
        deferredException = MoreObjects.firstNonNull(deferredException, e);
        continue;
      }
      if (cachedValue == null) {
        Preconditions.checkState(
            env.valuesMissing(), "no starlark import value for %s", importLookupKey);
        // We continue making inline calls even if some requested values are missing, to maximize
        // the number of dependent (non-inlined) SkyFunctions that are requested, thus avoiding a
        // quadratic number of restarts.
        valuesMissing = true;
      } else {
        SkyValue skyValue = cachedValue.getValue();
        skylarkImportMap.put(importLookupKey, skyValue);
        inlineCachedValueBuilder.addTransitiveDeps(cachedValue);
      }
    }
    if (deferredException != null) {
      Throwables.throwIfInstanceOf(deferredException, SkylarkImportFailedException.class);
      Throwables.throwIfInstanceOf(deferredException, InconsistentFilesystemException.class);
      throw new IllegalStateException(
          "caught a checked exception of unexpected type", deferredException);
    }
    return valuesMissing ? null : skylarkImportMap;
  }

  /** Creates the Extension to be imported. */
  private Extension createExtension(
      StarlarkFile file,
      Label extensionLabel,
      Map<String, Extension> importMap,
      StarlarkSemantics starlarkSemantics,
      Environment env,
      boolean inWorkspace,
      ImmutableMap<RepositoryName, RepositoryName> repositoryMapping)
      throws SkylarkImportFailedException, InterruptedException {
    StoredEventHandler eventHandler = new StoredEventHandler();
    // Any change to an input file may affect program behavior,
    // even if only by changing line numbers in error messages.
    PathFragment extensionFile = extensionLabel.toPathFragment();
    try (Mutability mutability = Mutability.create("importing", extensionFile)) {
      StarlarkThread thread =
          ruleClassProvider.createRuleClassStarlarkThread(
              extensionLabel,
              mutability,
              starlarkSemantics,
              StarlarkThread.makeDebugPrintHandler(eventHandler),
              file.getContentHashCode(),
              importMap,
              packageFactory.getNativeModule(inWorkspace),
              repositoryMapping);
      execAndExport(file, extensionLabel, eventHandler, thread);

      Event.replayEventsOn(env.getListener(), eventHandler.getEvents());
      for (Postable post : eventHandler.getPosts()) {
        env.getListener().post(post);
      }
      if (eventHandler.hasErrors()) {
        throw SkylarkImportFailedException.errors(extensionFile);
      }
      return new Extension(thread);
    }
  }

  // Precondition: file is validated and error-free.
  public static void execAndExport(
      StarlarkFile file, Label extensionLabel, EventHandler handler, StarlarkThread thread)
      throws InterruptedException {

    // Intercept execution after every assignment at top level
    // and "export" any newly assigned exportable globals.
    // TODO(adonovan): change the semantics; see b/65374671.
    thread.setPostAssignHook(
        (name, value) -> {
          if (value instanceof SkylarkExportable) {
            SkylarkExportable exp = (SkylarkExportable) value;
            if (!exp.isExported()) {
              try {
                exp.export(extensionLabel, name);
              } catch (EvalException ex) {
                handler.handle(Event.error(ex.getLocation(), ex.getMessage()));
              }
            }
          }
        });

    try {
      EvalUtils.exec(file, thread);
    } catch (EvalException ex) {
      handler.handle(Event.error(ex.getLocation(), ex.getMessage()));
    }
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  static final class SkylarkImportFailedException extends Exception {
    private SkylarkImportFailedException(String errorMessage) {
      super(errorMessage);
    }

    private SkylarkImportFailedException(String errorMessage, Exception cause) {
      super(errorMessage, cause);
    }

    static SkylarkImportFailedException errors(PathFragment file) {
      return new SkylarkImportFailedException(
          String.format("Extension file '%s' has errors", file));
    }

    static SkylarkImportFailedException errorReadingFile(
        PathFragment file, ErrorReadingSkylarkExtensionException cause) {
      return new SkylarkImportFailedException(
          String.format(
              "Encountered error while reading extension file '%s': %s",
              file,
              cause.getMessage()),
          cause);
    }

    static SkylarkImportFailedException noBuildFile(Label file, @Nullable String reason) {
      if (reason != null) {
        return new SkylarkImportFailedException(
            String.format("Unable to find package for %s: %s.", file, reason));
      }
      return new SkylarkImportFailedException(
          String.format("Every .bzl file must have a corresponding package, but '%s' "
              + "does not have one. Please create a BUILD file in the same or any parent directory."
              + " Note that this BUILD file does not need to do anything except exist.", file));
    }

    static SkylarkImportFailedException labelCrossesPackageBoundary(
        Label fileLabel,
        ContainingPackageLookupValue containingPackageLookupValue) {
      return new SkylarkImportFailedException(
          ContainingPackageLookupValue.getErrorMessageForLabelCrossingPackageBoundary(
              // We don't actually know the proper Root to pass in here (since we don't e.g. know
              // the root of the bzl/BUILD file that is trying to load 'fileLabel'). Therefore we
              // just pass in the Root of the containing package in order to still get a useful
              // error message for the user.
              containingPackageLookupValue.getContainingPackageRoot(),
              fileLabel,
              containingPackageLookupValue));
    }

    static SkylarkImportFailedException skylarkErrors(PathFragment file) {
      return new SkylarkImportFailedException(String.format("Extension '%s' has errors", file));
    }
  }

  private static final class SkylarkImportLookupFunctionException extends SkyFunctionException {
    private SkylarkImportLookupFunctionException(SkylarkImportFailedException cause) {
      super(cause, Transience.PERSISTENT);
    }

    private SkylarkImportLookupFunctionException(InconsistentFilesystemException e,
        Transience transience) {
      super(e, transience);
    }
  }
}
