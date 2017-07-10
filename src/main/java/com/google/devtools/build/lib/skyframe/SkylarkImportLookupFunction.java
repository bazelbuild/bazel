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

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.SkylarkExportable;
import com.google.devtools.build.lib.skyframe.SkylarkImportLookupValue.SkylarkImportLookupKey;
import com.google.devtools.build.lib.syntax.AssignmentStatement;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Environment.Extension;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.LoadStatement;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.SkylarkImport;
import com.google.devtools.build.lib.syntax.SkylarkSemanticsOptions;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException2;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import javax.annotation.Nullable;

/**
 * A Skyframe function to look up and import a single Skylark extension.
 *
 * <p> Given a {@link Label} referencing a Skylark file, attempts to locate the file and load it.
 * The Label must be absolute, and must not reference the special {@code external} package. If
 * loading is successful, returns a {@link SkylarkImportLookupValue} that encapsulates
 * the loaded {@link Extension} and {@link SkylarkFileDependency} information. If loading is
 * unsuccessful, throws a {@link SkylarkImportLookupFunctionException} that encapsulates the
 * cause of the failure.
 */
public class SkylarkImportLookupFunction implements SkyFunction {

  private final RuleClassProvider ruleClassProvider;
  private final PackageFactory packageFactory;

  public SkylarkImportLookupFunction(
    RuleClassProvider ruleClassProvider, PackageFactory packageFactory) {
    this.ruleClassProvider = ruleClassProvider;
    this.packageFactory = packageFactory;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException,
      InterruptedException {
    SkylarkImportLookupKey key = (SkylarkImportLookupKey) skyKey.argument();
    try {
      return computeInternal(key.importLabel, key.inWorkspace, env, null);
    } catch (InconsistentFilesystemException e) {
      throw new SkylarkImportLookupFunctionException(e, Transience.PERSISTENT);
    } catch (SkylarkImportFailedException e) {
      throw new SkylarkImportLookupFunctionException(e);
    }
  }

  SkyValue computeWithInlineCalls(
      SkyKey skyKey, Environment env, LinkedHashMap<Label, SkylarkImportLookupValue> visited)
      throws InconsistentFilesystemException, SkylarkImportFailedException, InterruptedException {
    return computeWithInlineCallsInternal(skyKey, env, visited);
  }

  private SkyValue computeWithInlineCallsInternal(
      SkyKey skyKey, Environment env, LinkedHashMap<Label, SkylarkImportLookupValue> visited)
      throws InconsistentFilesystemException, SkylarkImportFailedException, InterruptedException {
    SkylarkImportLookupKey key = (SkylarkImportLookupKey) skyKey.argument();
    SkylarkImportLookupValue precomputedResult = visited.get(key.importLabel);
    if (precomputedResult != null) {
      return precomputedResult;
    }
    return computeInternal(
        key.importLabel,
        key.inWorkspace,
        env,
        Preconditions.checkNotNull(visited, key.importLabel));
  }

  private SkyValue computeInternal(
      Label fileLabel,
      boolean inWorkspace,
      Environment env,
      @Nullable LinkedHashMap<Label, SkylarkImportLookupValue> alreadyVisited)
      throws InconsistentFilesystemException, SkylarkImportFailedException, InterruptedException {
    PathFragment filePath = fileLabel.toPathFragment();

    SkylarkSemanticsOptions skylarkSemantics = PrecomputedValue.SKYLARK_SEMANTICS.get(env);
    if (skylarkSemantics == null) {
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
      throw SkylarkImportFailedException.noFile(astLookupValue.getErrorMsg());
    }
    BuildFileAST ast = astLookupValue.getAST();
    if (ast.containsErrors()) {
      throw SkylarkImportFailedException.skylarkErrors(filePath);
    }

    // Process the load statements in the file.
    ImmutableList<SkylarkImport> imports = ast.getImports();
    Map<String, Extension> extensionsForImports = Maps.newHashMapWithExpectedSize(imports.size());
    ImmutableList.Builder<SkylarkFileDependency> fileDependencies = ImmutableList.builder();
    ImmutableMap<String, Label> labelsForImports;

    // Find the labels corresponding to the load statements.
    labelsForImports = findLabelsForLoadStatements(imports, fileLabel, env);
    if (labelsForImports == null) {
      return null;
    }

    // Look up and load the imports.
    ImmutableCollection<Label> importLabels = labelsForImports.values();
    List<SkyKey> importLookupKeys =
        Lists.newArrayListWithExpectedSize(importLabels.size());
    for (Label importLabel : importLabels) {
      importLookupKeys.add(SkylarkImportLookupValue.key(importLabel, inWorkspace));
    }
    Map<SkyKey, SkyValue> skylarkImportMap;
    boolean valuesMissing = false;
    if (alreadyVisited == null) {
      // Not inlining.
      skylarkImportMap = env.getValues(importLookupKeys);
      valuesMissing = env.valuesMissing();
    } else {
      // Inlining calls to SkylarkImportLookupFunction.
      if (alreadyVisited.containsKey(fileLabel)) {
        ImmutableList<Label> cycle =
            CycleUtils.splitIntoPathAndChain(Predicates.equalTo(fileLabel), alreadyVisited.keySet())
                .second;
        throw new SkylarkImportFailedException("Skylark import cycle: " + cycle);
      }
      alreadyVisited.put(fileLabel, null);
      skylarkImportMap = Maps.newHashMapWithExpectedSize(imports.size());
      for (SkyKey importLookupKey : importLookupKeys) {
        SkyValue skyValue =
            this.computeWithInlineCallsInternal(importLookupKey, env, alreadyVisited);
        if (skyValue == null) {
          Preconditions.checkState(
              env.valuesMissing(), "no skylark import value for %s", importLookupKey);
          // We continue making inline calls even if some requested values are missing, to maximize
          // the number of dependent (non-inlined) SkyFunctions that are requested, thus avoiding a
          // quadratic number of restarts.
          valuesMissing = true;
        } else {
          skylarkImportMap.put(importLookupKey, skyValue);
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
    for (Entry<String, Label> importEntry : labelsForImports.entrySet()) {
      String importString = importEntry.getKey();
      Label importLabel = importEntry.getValue();
      SkyKey keyForLabel = SkylarkImportLookupValue.key(importLabel, inWorkspace);
      SkylarkImportLookupValue importLookupValue =
          (SkylarkImportLookupValue) skylarkImportMap.get(keyForLabel);
      extensionsForImports.put(importString, importLookupValue.getEnvironmentExtension());
      fileDependencies.add(importLookupValue.getDependency());
    }

    // Skylark UserDefinedFunction-s in that file will share this function definition Environment,
    // which will be frozen by the time it is returned by createExtension.
    Extension extension = createExtension(
        ast,
        fileLabel,
        extensionsForImports,
        skylarkSemantics,
        env,
        inWorkspace);
    SkylarkImportLookupValue result =
        new SkylarkImportLookupValue(
            extension, new SkylarkFileDependency(fileLabel, fileDependencies.build()));
    if (alreadyVisited != null) {
      alreadyVisited.put(fileLabel, result);
    }
    return result;
  }

  /**
   * Computes the set of Labels corresponding to a collection of PathFragments representing absolute
   * import paths.
   *
   * @return a map from the computed {@link Label}s to the corresponding {@link PathFragment}s;
   *     {@code null} if any Skyframe dependencies are unavailable.
   * @throws SkylarkImportFailedException
   */
  @Nullable
  static ImmutableMap<PathFragment, Label> labelsForAbsoluteImports(
      ImmutableSet<PathFragment> pathsToLookup, Environment env)
      throws SkylarkImportFailedException, InterruptedException {

    // Import PathFragments are absolute, so there is a 1-1 mapping from corresponding Labels.
    ImmutableMap.Builder<PathFragment, Label> outputMap = new ImmutableMap.Builder<>();

    // The SkyKey here represents the directory containing an import PathFragment, hence there
    // can in general be multiple imports per lookup.
    Multimap<SkyKey, PathFragment> lookupMap = LinkedHashMultimap.create();
    for (PathFragment importPath : pathsToLookup) {
      PathFragment relativeImportPath = importPath.toRelative();
      PackageIdentifier pkgToLookUp =
          PackageIdentifier.createInMainRepo(relativeImportPath.getParentDirectory());
      lookupMap.put(ContainingPackageLookupValue.key(pkgToLookUp), importPath);
    }

    // Attempt to find a package for every directory containing an import.
    Map<SkyKey,
        ValueOrException2<BuildFileNotFoundException,
            InconsistentFilesystemException>> lookupResults =
        env.getValuesOrThrow(
            lookupMap.keySet(),
            BuildFileNotFoundException.class,
            InconsistentFilesystemException.class);
    if (env.valuesMissing()) {
     return null;
    }
    try {
      // Process lookup results.
      for (Entry<SkyKey,
               ValueOrException2<BuildFileNotFoundException,
                   InconsistentFilesystemException>> entry : lookupResults.entrySet()) {
        ContainingPackageLookupValue lookupValue =
            (ContainingPackageLookupValue) entry.getValue().get();
        if (!lookupValue.hasContainingPackage()) {
          // Although multiple imports may be in the same package-less directory, we only
          // report an error for the first one.
          PackageIdentifier lookupKey = ((PackageIdentifier) entry.getKey().argument());
          PathFragment importFile = lookupKey.getPackageFragment();
          throw SkylarkImportFailedException.noBuildFile(importFile);
        }
        PackageIdentifier pkgIdForImport = lookupValue.getContainingPackageName();
        PathFragment containingPkgPath = pkgIdForImport.getPackageFragment();
        for (PathFragment importPath : lookupMap.get(entry.getKey())) {
         PathFragment relativeImportPath = importPath.toRelative();
         String targetNameForImport = relativeImportPath.relativeTo(containingPkgPath).toString();
         try {
           outputMap.put(importPath, Label.create(pkgIdForImport, targetNameForImport));
         } catch (LabelSyntaxException e) {
           // While the Label is for the most part guaranteed to be well-formed by construction, an
           // error is still possible if the filename itself is malformed, e.g., contains control
           // characters. Since we expect this error to be very rare, for code simplicity, we allow
           // the error message to refer to a Label even though the filename was specified via a
           // simple path.
           throw new SkylarkImportFailedException(e);
         }
        }
      }
    } catch (BuildFileNotFoundException e) {
      // Thrown when there are IO errors looking for BUILD files.
      throw new SkylarkImportFailedException(e);
    } catch (InconsistentFilesystemException e) {
      throw new SkylarkImportFailedException(e);
    }

  return outputMap.build();
  }

  /**
   * Computes the set of {@link Label}s corresponding to a set of Skylark {@link LoadStatement}s.
   *
   * @param imports a collection of Skylark {@link LoadStatement}s
   * @param containingFileLabel the {@link Label} of the file containing the load statements
   * @return an {@link ImmutableMap} which maps a {@link String} used in the load statement to its
   *     corresponding {@Label}. Returns {@code null} if any Skyframe dependencies are unavailable.
   * @throws SkylarkImportFailedException if no package can be found that contains the loaded file
   */
  @Nullable
  static ImmutableMap<String, Label> findLabelsForLoadStatements(
      ImmutableCollection<SkylarkImport> imports, Label containingFileLabel, Environment env)
      throws SkylarkImportFailedException, InterruptedException {
    Preconditions.checkArgument(
        !containingFileLabel.getPackageIdentifier().getRepository().isDefault());
    Map<String, Label> outputMap = Maps.newHashMapWithExpectedSize(imports.size());

    // Filter relative vs. absolute paths.
    ImmutableSet.Builder<PathFragment> absoluteImportsToLookup = new ImmutableSet.Builder<>();
    // We maintain a multimap from path fragments to their correspond import strings, to cover the
    // (unlikely) case where two distinct import strings generate the same path fragment.
    ImmutableMultimap.Builder<PathFragment, String> pathToImports =
        new ImmutableMultimap.Builder<>();
    for (SkylarkImport imp : imports) {
      if (imp.hasAbsolutePath()) {
        absoluteImportsToLookup.add(imp.getAbsolutePath());
        pathToImports.put(imp.getAbsolutePath(), imp.getImportString());
      } else {
        outputMap.put(imp.getImportString(), imp.getLabel(containingFileLabel));
      }
    }

    // Look up labels for absolute paths.
    ImmutableMap<PathFragment, Label> absoluteLabels =
        labelsForAbsoluteImports(absoluteImportsToLookup.build(), env);
    if (absoluteLabels == null) {
      return null;
    }
    for (Entry<PathFragment, Label> entry : absoluteLabels.entrySet()) {
      PathFragment currPath = entry.getKey();
      Label currLabel = entry.getValue();
      for (String importString : pathToImports.build().get(currPath)) {
        outputMap.put(importString, currLabel);
      }
    }

    ImmutableMap<String, Label> immutableOutputMap = ImmutableMap.copyOf(outputMap);
    return immutableOutputMap;
  }

  /**
   * Creates the Extension to be imported.
   */
  private Extension createExtension(
      BuildFileAST ast,
      Label extensionLabel,
      Map<String, Extension> importMap,
      SkylarkSemanticsOptions skylarkSemantics,
      Environment env,
      boolean inWorkspace)
      throws SkylarkImportFailedException, InterruptedException {
    StoredEventHandler eventHandler = new StoredEventHandler();
    // TODO(bazel-team): this method overestimates the changes which can affect the
    // Skylark RuleClass. For example changes to comments or unused functions can modify the hash.
    // A more accurate - however much more complicated - way would be to calculate a hash based on
    // the transitive closure of the accessible AST nodes.
    PathFragment extensionFile = extensionLabel.toPathFragment();
    try (Mutability mutability = Mutability.create("importing %s", extensionFile)) {
      com.google.devtools.build.lib.syntax.Environment extensionEnv =
          ruleClassProvider
              .createSkylarkRuleClassEnvironment(
                  extensionLabel, mutability, skylarkSemantics,
                  eventHandler, ast.getContentHashCode(), importMap)
              .setupOverride("native", packageFactory.getNativeModule(inWorkspace));
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
    ImmutableSet<String> boundNames = assignmentStatement.getLValue().boundNames();
    for (String name : boundNames) {
      Object lookup = extensionEnv.lookup(name);
      if (lookup instanceof SkylarkExportable) {
        try {
          SkylarkExportable exportable = (SkylarkExportable) lookup;
          if (!exportable.isExported()) {
            exportable.export(extensionLabel, name);
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

    private SkylarkImportFailedException(InconsistentFilesystemException e) {
      this(e.getMessage(), e);
    }

    private SkylarkImportFailedException(BuildFileNotFoundException e) {
      this(e.getMessage(), e);
    }

    private SkylarkImportFailedException(LabelSyntaxException e) {
      this(e.getMessage(), e);
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

    static SkylarkImportFailedException noFile(String reason) {
      return new SkylarkImportFailedException(
          String.format("Extension file not found. %s", reason));
    }

    static SkylarkImportFailedException noBuildFile(PathFragment file) {
      return new SkylarkImportFailedException(
          String.format("Every .bzl file must have a corresponding package, but '%s' "
              + "does not have one. Please create a BUILD file in the same or any parent directory."
              + " Note that this BUILD file does not need to do anything except exist.", file));
    }

    static SkylarkImportFailedException skylarkErrors(PathFragment file) {
      return new SkylarkImportFailedException(String.format("Extension '%s' has errors", file));
    }

   static final String NO_EXT_WORKSPACE_LOAD_MSG_TEMPLATE =
       "Extension file '%s' may not be loaded from a WORKSPACE file "
           + "since the extension file is located in an external repository.";
   static SkylarkImportFailedException noExternalLoadsFromWorkspace(Label fileLabel) {
      return new SkylarkImportFailedException(String.format(NO_EXT_WORKSPACE_LOAD_MSG_TEMPLATE,
          fileLabel));
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

    private SkylarkImportLookupFunctionException(BuildFileNotFoundException e,
        Transience transience) {
      super(e, transience);
    }
  }
}
