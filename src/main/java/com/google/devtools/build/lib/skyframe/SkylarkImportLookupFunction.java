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

import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.InconsistentFilesystemException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.SkylarkExportable;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.skyframe.SkylarkImportLookupValue.SkylarkImportLookupKey;
import com.google.devtools.build.lib.syntax.AssignmentStatement;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Environment.Extension;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Identifier;
import com.google.devtools.build.lib.syntax.LoadStatement;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.SkylarkImport;
import com.google.devtools.build.lib.syntax.SkylarkImports;
import com.google.devtools.build.lib.syntax.SkylarkImports.SkylarkImportSyntaxException;
import com.google.devtools.build.lib.syntax.SkylarkSemantics;
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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
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
  private Cache<SkyKey, CachedSkylarkImportLookupValueAndDeps> skylarkImportLookupValueCache;

  private static final Logger logger =
      Logger.getLogger(SkylarkImportLookupFunction.class.getName());

  public SkylarkImportLookupFunction(
    RuleClassProvider ruleClassProvider, PackageFactory packageFactory) {
    this.ruleClassProvider = ruleClassProvider;
    this.packageFactory = packageFactory;
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
          /*alreadyVisited=*/ null,
          /*inlineCachedValueBuilder=*/ null);
    } catch (InconsistentFilesystemException e) {
      throw new SkylarkImportLookupFunctionException(e, Transience.PERSISTENT);
    } catch (SkylarkImportFailedException e) {
      throw new SkylarkImportLookupFunctionException(e);
    }
  }

  @Nullable
  SkylarkImportLookupValue computeWithInlineCalls(
      SkyKey skyKey, Environment env, int expectedSizeOfVisitedSet)
      throws InconsistentFilesystemException, SkylarkImportFailedException, InterruptedException {
    // We use the visited set to track if there are any cyclic dependencies when loading the
    // skylark file.
    LinkedHashMap<Label, CachedSkylarkImportLookupValueAndDeps> visited =
        Maps.newLinkedHashMapWithExpectedSize(expectedSizeOfVisitedSet);
    CachedSkylarkImportLookupValueAndDeps cachedSkylarkImportLookupValueAndDeps =
        computeWithInlineCallsInternal(skyKey, env, visited);
    if (cachedSkylarkImportLookupValueAndDeps == null) {
      return null;
    }
    cachedSkylarkImportLookupValueAndDeps.traverse(env::registerDependencies);
    return cachedSkylarkImportLookupValueAndDeps.getValue();
  }

  @Nullable
  private CachedSkylarkImportLookupValueAndDeps computeWithInlineCallsInternal(
      SkyKey skyKey,
      Environment env,
      LinkedHashMap<Label, CachedSkylarkImportLookupValueAndDeps> visited)
      throws InconsistentFilesystemException, SkylarkImportFailedException, InterruptedException {
    SkylarkImportLookupKey key = (SkylarkImportLookupKey) skyKey.argument();
    CachedSkylarkImportLookupValueAndDeps precomputedResult = visited.get(key.importLabel);
    if (precomputedResult != null) {
      // We have already registered all the deps for this value.
      return precomputedResult;
    }
    // Note that we can't block other threads on the computation of this value due to a potential
    // deadlock on a cycle. Although we are repeating some work, it is possible we have an import
    // cycle where one thread starts at one side of the cycle and the other thread starts at the
    // other side, and they then wait forever on the results of each others computations.
    CachedSkylarkImportLookupValueAndDeps cachedSkylarkImportLookupValueAndDeps =
        skylarkImportLookupValueCache.getIfPresent(skyKey);
    if (cachedSkylarkImportLookupValueAndDeps != null) {
      return cachedSkylarkImportLookupValueAndDeps;
    }

    CachedSkylarkImportLookupValueAndDeps.Builder inlineCachedValueBuilder =
        CachedSkylarkImportLookupValueAndDeps.newBuilder();
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
            key.importLabel,
            key.inWorkspace,
            key.workspaceChunk,
            key.workspacePath,
            recordingEnv,
            Preconditions.checkNotNull(visited, key.importLabel),
            inlineCachedValueBuilder);

    if (value != null) {
      inlineCachedValueBuilder.setValue(value);
      cachedSkylarkImportLookupValueAndDeps = inlineCachedValueBuilder.build();
      skylarkImportLookupValueCache.put(skyKey, cachedSkylarkImportLookupValueAndDeps);
      visited.put(key.importLabel, cachedSkylarkImportLookupValueAndDeps);
    }
    return cachedSkylarkImportLookupValueAndDeps;
  }

  public void resetCache() {
    if (skylarkImportLookupValueCache != null) {
      logger.info(
          "Starlark inlining cache stats from earlier build: "
              + skylarkImportLookupValueCache.stats());
    }
    skylarkImportLookupValueCache =
        CacheBuilder.newBuilder()
            .concurrencyLevel(BlazeInterners.concurrencyLevel())
            .maximumSize(10000)
            .recordStats()
            .build();
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
      @Nullable LinkedHashMap<Label, CachedSkylarkImportLookupValueAndDeps> alreadyVisited,
      @Nullable CachedSkylarkImportLookupValueAndDeps.Builder inlineCachedValueBuilder)
      throws InconsistentFilesystemException, SkylarkImportFailedException, InterruptedException {
    PathFragment filePath = fileLabel.toPathFragment();

    SkylarkSemantics skylarkSemantics = PrecomputedValue.SKYLARK_SEMANTICS.get(env);
    if (skylarkSemantics == null) {
      return null;
    }

    if (skylarkSemantics.incompatibleDisallowLoadLabelsToCrossPackageBoundaries()) {
      PathFragment dir = Label.getContainingDirectory(fileLabel);
      PackageIdentifier dirId =
          PackageIdentifier.create(fileLabel.getPackageIdentifier().getRepository(), dir);
      ContainingPackageLookupValue containingPackageLookupValue;
      try {
        containingPackageLookupValue = (ContainingPackageLookupValue) env.getValueOrThrow(
            ContainingPackageLookupValue.key(dirId),
            BuildFileNotFoundException.class,
            InconsistentFilesystemException.class);
      } catch (BuildFileNotFoundException e) {
        throw SkylarkImportFailedException.errorReadingFile(
            fileLabel.toPathFragment(),
            new ErrorReadingSkylarkExtensionException(e));
      }
      if (containingPackageLookupValue == null) {
        return null;
      }
      if (!containingPackageLookupValue.hasContainingPackage()) {
        throw SkylarkImportFailedException.noBuildFile(fileLabel.toPathFragment());
      }
      if (!containingPackageLookupValue.getContainingPackageName().equals(
          fileLabel.getPackageIdentifier())) {
        throw SkylarkImportFailedException.labelCrossesPackageBoundary(
            fileLabel, containingPackageLookupValue);
      }
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
    BuildFileAST ast = astLookupValue.getAST();
    if (ast.containsErrors()) {
      throw SkylarkImportFailedException.skylarkErrors(filePath);
    }

    // Process the load statements in the file.
    ImmutableList<SkylarkImport> unRemappedImports = ast.getImports();
    ImmutableMap<RepositoryName, RepositoryName> repositoryMapping =
        getRepositoryMapping(workspaceChunk, workspacePath, fileLabel, env);

    if (repositoryMapping == null) {
      return null;
    }

    ImmutableList<SkylarkImport> imports =
        remapImports(unRemappedImports, workspaceChunk, repositoryMapping);

    ImmutableMap<String, Label> labelsForImports = getLabelsForLoadStatements(imports, fileLabel);
    ImmutableCollection<Label> importLabels = labelsForImports.values();

    // Look up and load the imports.
    List<SkyKey> importLookupKeys =
        Lists.newArrayListWithExpectedSize(labelsForImports.size());
    for (Label importLabel : importLabels) {
      if (inWorkspace) {
        importLookupKeys.add(
            SkylarkImportLookupValue.keyInWorkspace(importLabel, workspaceChunk, workspacePath));
      } else {
        importLookupKeys.add(SkylarkImportLookupValue.key(importLabel));
      }
    }
    Map<SkyKey, SkyValue> skylarkImportMap;
    boolean valuesMissing = false;
    if (alreadyVisited == null) {
      // Not inlining.

      Map<SkyKey, ValueOrException<SkylarkImportFailedException>> values =
          env.getValuesOrThrow(importLookupKeys, SkylarkImportFailedException.class);
      skylarkImportMap = Maps.newHashMapWithExpectedSize(values.size());
      for (Map.Entry<SkyKey, ValueOrException<SkylarkImportFailedException>> entry :
          env.getValuesOrThrow(importLookupKeys, SkylarkImportFailedException.class).entrySet()) {
        try {
          skylarkImportMap.put(entry.getKey(), entry.getValue().get());
        } catch (SkylarkImportFailedException exn) {
          throw new SkylarkImportFailedException(
              "in " + ast.getLocation().getPath() + ": " + exn.getMessage());
        }
      }
      valuesMissing = env.valuesMissing();
    } else {
      Preconditions.checkNotNull(
          inlineCachedValueBuilder,
          "Expected inline cached value builder to be not-null when inlining.");
      // Inlining calls to SkylarkImportLookupFunction.
      if (alreadyVisited.containsKey(fileLabel)) {
        ImmutableList<Label> cycle =
            CycleUtils.splitIntoPathAndChain(Predicates.equalTo(fileLabel), alreadyVisited.keySet())
                .second;
        throw new SkylarkImportFailedException("Starlark import cycle: " + cycle);
      }
      alreadyVisited.put(fileLabel, null);
      skylarkImportMap = Maps.newHashMapWithExpectedSize(imports.size());

      Preconditions.checkState(
          env instanceof RecordingSkyFunctionEnvironment,
          "Expected to be recording dep requests when inlining SkylarkImportLookupFunction: %s",
          fileLabel);
      Environment strippedEnv = ((RecordingSkyFunctionEnvironment) env).getDelegate();
      for (SkyKey importLookupKey : importLookupKeys) {
        CachedSkylarkImportLookupValueAndDeps cachedValue =
            this.computeWithInlineCallsInternal(importLookupKey, strippedEnv, alreadyVisited);
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
      // All imports traversed, this key can no longer be part of a cycle.
      Preconditions.checkState(alreadyVisited.remove(fileLabel) == null, fileLabel);
    }
    if (valuesMissing) {
      // This means some imports are unavailable.
      return null;
    }

    // Process the loaded imports.
    Map<String, Extension> extensionsForImports = Maps.newHashMapWithExpectedSize(imports.size());
    ImmutableList.Builder<SkylarkFileDependency> fileDependencies =
        ImmutableList.builderWithExpectedSize(importLabels.size());
    for (Map.Entry<String, Label> importEntry : labelsForImports.entrySet()) {
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
            ast,
            fileLabel,
            extensionsForImports,
            skylarkSemantics,
            env,
            inWorkspace,
            repositoryMapping);
    SkylarkImportLookupValue result =
        new SkylarkImportLookupValue(
            extension, new SkylarkFileDependency(fileLabel, fileDependencies.build()));
    return result;
  }

  private ImmutableMap<RepositoryName, RepositoryName> getRepositoryMapping(
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
   * This method takes in a list of {@link SkylarkImport}s (load statements) as they appear in the
   * BUILD, bzl, or WORKSPACE file they originated from and optionally remaps the load statements
   * using the repository mappings provided in the WORKSPACE file.
   *
   * <p>If the {@link SkylarkImport}s originated from a WORKSPACE file, then the repository mappings
   * are pulled from the previous {@link WorkspaceFileValue}. If they didn't originate from a
   * WORKSPACE file then the repository mappings are pulled from the fully computed {@link
   * RepositoryMappingValue}.
   *
   * <p>There is a chance that SkyValues requested are not yet computed and so SkyFunction callers
   * of this method need to check if the return value is null and then return null themselves.
   *
   * @param unRemappedImports the list of load statements to be remapped
   * @param workspaceChunk the workspaceChunk we are currently evaluating that this load statement
   *     originated from. WORKSPACE files are chunked at every non-consecutive load statement and
   *     evaluated separately. See {@link WorkspaceFileValue} for more information.
   * @param repositoryMapping map from original repository names to new repository names given
   *     by the main repository
   * @return a list of remapped {@link SkylarkImport}s or null if any SkyValue requested wasn't
   *     fully computed yet
   * @throws InterruptedException
   */
  private ImmutableList<SkylarkImport> remapImports(
      ImmutableList<SkylarkImport> unRemappedImports,
      int workspaceChunk,
      ImmutableMap<RepositoryName, RepositoryName> repositoryMapping) {

    // There is no previous workspace chunk
    if (workspaceChunk == 0) {
      return unRemappedImports;
    }

    ImmutableList.Builder<SkylarkImport> builder = ImmutableList.builder();
    for (SkylarkImport notRemappedImport : unRemappedImports) {
      try {
        SkylarkImport newImport =
            SkylarkImports.create(notRemappedImport.getImportString(), repositoryMapping);
        builder.add(newImport);
      } catch (SkylarkImportSyntaxException ignored) {
        // This won't happen because we are constructing a SkylarkImport from a SkylarkImport so
        // it must be valid
        throw new AssertionError("SkylarkImportSyntaxException", ignored);
      }
    }
    return builder.build();
  }

  /**
   * Given a collection of {@link SkylarkImport}, returns a map from import string to label of
   * imported file.
   *
   * @param imports a collection of Skylark {@link LoadStatement}s
   * @param containingFileLabel the {@link Label} of the file containing the load statements
   */
  @Nullable
  static ImmutableMap<String, Label> getLabelsForLoadStatements(
      ImmutableCollection<SkylarkImport> imports, Label containingFileLabel) {
    Preconditions.checkArgument(
        !containingFileLabel.getPackageIdentifier().getRepository().isDefault());
    return imports.stream()
        .collect(
            toImmutableMap(
                SkylarkImport::getImportString,
                imp -> imp.getLabel(containingFileLabel),
                (oldLabel, newLabel) -> oldLabel));
  }

  /** Creates the Extension to be imported. */
  private Extension createExtension(
      BuildFileAST ast,
      Label extensionLabel,
      Map<String, Extension> importMap,
      SkylarkSemantics skylarkSemantics,
      Environment env,
      boolean inWorkspace,
      ImmutableMap<RepositoryName, RepositoryName> repositoryMapping)
      throws SkylarkImportFailedException, InterruptedException {
    StoredEventHandler eventHandler = new StoredEventHandler();
    // TODO(bazel-team): this method overestimates the changes which can affect the
    // Skylark RuleClass. For example changes to comments or unused functions can modify the hash.
    // A more accurate - however much more complicated - way would be to calculate a hash based on
    // the transitive closure of the accessible AST nodes.
    PathFragment extensionFile = extensionLabel.toPathFragment();
    try (Mutability mutability = Mutability.create("importing %s", extensionFile)) {
      com.google.devtools.build.lib.syntax.Environment extensionEnv =
          ruleClassProvider.createSkylarkRuleClassEnvironment(
              extensionLabel,
              mutability,
              skylarkSemantics,
              eventHandler,
              ast.getContentHashCode(),
              importMap,
              repositoryMapping);
      extensionEnv.setupOverride("native", packageFactory.getNativeModule(inWorkspace));
      execAndExport(ast, extensionLabel, eventHandler, extensionEnv);

      Event.replayEventsOn(env.getListener(), eventHandler.getEvents());
      for (Postable post : eventHandler.getPosts()) {
        env.getListener().post(post);
      }
      if (eventHandler.hasErrors()) {
        throw SkylarkImportFailedException.errors(extensionFile);
      }
      return new Extension(extensionEnv);
    }
  }

  public static void execAndExport(BuildFileAST ast, Label extensionLabel,
      EventHandler eventHandler,
      com.google.devtools.build.lib.syntax.Environment extensionEnv) throws InterruptedException {
    ImmutableList<Statement> statements = ast.getStatements();
    for (Statement statement : statements) {
      ast.execTopLevelStatement(statement, extensionEnv, eventHandler);
      possiblyExport(statement, extensionLabel, eventHandler, extensionEnv);
    }
  }

  private static void possiblyExport(Statement statement, Label extensionLabel,
      EventHandler eventHandler,
      com.google.devtools.build.lib.syntax.Environment extensionEnv) {
    if (!(statement instanceof AssignmentStatement)) {
      return;
    }
    AssignmentStatement assignmentStatement = (AssignmentStatement) statement;
    ImmutableSet<Identifier> boundIdentifiers = assignmentStatement.getLValue().boundIdentifiers();
    for (Identifier ident : boundIdentifiers) {
      Object lookup = extensionEnv.moduleLookup(ident.getName());
      if (lookup instanceof SkylarkExportable) {
        try {
          SkylarkExportable exportable = (SkylarkExportable) lookup;
          if (!exportable.isExported()) {
            exportable.export(extensionLabel, ident.getName());
          }
        } catch (EvalException e) {
          eventHandler.handle(Event.error(e.getLocation(), e.getMessage()));
        }
      }
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

    static SkylarkImportFailedException noBuildFile(PathFragment file) {
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
