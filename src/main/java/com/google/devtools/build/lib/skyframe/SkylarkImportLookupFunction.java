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

import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.rules.SkylarkRuleClassFunctions;
import com.google.devtools.build.lib.skyframe.ASTFileLookupValue.ASTLookupInputException;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Environment.Extension;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * A Skyframe function to look up and import a single Skylark extension.
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
    return computeInternal(skyKey, env, null);
  }

  SkyValue computeWithInlineCalls(SkyKey skyKey, Environment env, Set<SkyKey> visitedKeysForCycle)
      throws SkyFunctionException, InterruptedException {
    return computeInternal(skyKey, env, Preconditions.checkNotNull(visitedKeysForCycle, skyKey));
  }

  SkyValue computeInternal(SkyKey skyKey, Environment env,
      @Nullable Set<SkyKey> visitedKeysForCycle)
      throws SkyFunctionException, InterruptedException {
    PackageIdentifier arg = (PackageIdentifier) skyKey.argument();
    PathFragment file = arg.getPackageFragment();
    ASTFileLookupValue astLookupValue = null;
    try {
      SkyKey astLookupKey = ASTFileLookupValue.key(arg);
      astLookupValue = (ASTFileLookupValue) env.getValueOrThrow(astLookupKey,
          ErrorReadingSkylarkExtensionException.class, InconsistentFilesystemException.class);
    } catch (ErrorReadingSkylarkExtensionException e) {
      throw new SkylarkImportLookupFunctionException(SkylarkImportFailedException.errorReadingFile(
          file, e.getMessage()));
    } catch (InconsistentFilesystemException e) {
      throw new SkylarkImportLookupFunctionException(e, Transience.PERSISTENT);
    }
    if (astLookupValue == null) {
      return null;
    }
    if (astLookupValue.getAST() == null) {
      // Skylark import files have to exist.
      throw new SkylarkImportLookupFunctionException(SkylarkImportFailedException.noFile(file));
    }

    BuildFileAST ast = astLookupValue.getAST();
    if (ast.containsErrors()) {
      throw new SkylarkImportLookupFunctionException(
          SkylarkImportFailedException.skylarkErrors(file));
    }

    Label label = pathFragmentToLabel(arg.getRepository(), file, env);
    if (label == null) {
      Preconditions.checkState(env.valuesMissing(), "null label with no missing %s", file);
      return null;
    }

    Map<Location, PathFragment> astImports = ast.getImports();
    Map<PathFragment, Extension> importMap = Maps.newHashMapWithExpectedSize(astImports.size());
    ImmutableList.Builder<SkylarkFileDependency> fileDependencies = ImmutableList.builder();
    Map<SkyKey, PathFragment> skylarkImports = Maps.newHashMapWithExpectedSize(astImports.size());
    for (Map.Entry<Location, PathFragment> entry : ast.getImports().entrySet()) {
      try {
        skylarkImports.put(
            PackageFunction.getImportKey(entry, ruleClassProvider.getPreludePath(), file, arg),
            entry.getValue());
      } catch (ASTLookupInputException e) {
        throw new SkylarkImportLookupFunctionException(e, Transience.PERSISTENT);
      }
    }

    Map<SkyKey, SkyValue> skylarkImportMap;
    boolean valuesMissing = false;
    if (visitedKeysForCycle == null) {
      // Not inlining.
      skylarkImportMap = env.getValues(skylarkImports.keySet());
      valuesMissing = env.valuesMissing();
    } else {
      // inlining calls to SkylarkImportLookupFunction.
      if (!visitedKeysForCycle.add(skyKey)) {
        ImmutableList<SkyKey> cycle =
            CycleUtils.splitIntoPathAndChain(Predicates.equalTo(skyKey), visitedKeysForCycle)
                .second;
        if (env.getValue(SkylarkImportUniqueCycleValue.key(cycle)) == null) {
          return null;
        }
        throw new SkylarkImportLookupFunctionException(
            new SkylarkImportFailedException("Skylark import cycle"));
      }
      skylarkImportMap = Maps.newHashMapWithExpectedSize(astImports.size());
      for (SkyKey skylarkImport : skylarkImports.keySet()) {
        SkyValue skyValue = this.computeWithInlineCalls(skylarkImport, env, visitedKeysForCycle);
        if (skyValue == null) {
          Preconditions.checkState(
              env.valuesMissing(), "no skylark import value for %s", skylarkImport);
          // Don't give up on computing. This is against the Skyframe contract, but we don't want to
          // pay the price of serializing all these calls, since they are fundamentally independent.
          valuesMissing = true;
        } else {
          skylarkImportMap.put(skylarkImport, skyValue);
        }
      }
      // All imports traversed, this key can no longer be part of a cycle.
      visitedKeysForCycle.remove(skyKey);
    }

    if (valuesMissing) {
      // This means some imports are unavailable.
      return null;
    }

    for (Map.Entry<SkyKey, SkyValue> entry : skylarkImportMap.entrySet()) {
      SkylarkImportLookupValue importLookupValue = (SkylarkImportLookupValue) entry.getValue();
      importMap.put(
          skylarkImports.get(entry.getKey()), importLookupValue.getEnvironmentExtension());
      fileDependencies.add(importLookupValue.getDependency());
    }

    // Skylark UserDefinedFunction-s in that file will share this function definition Environment,
    // which will be frozen by the time it is returned by createExtension.
    Extension extension = createExtension(ast, file, importMap, env);

    return new SkylarkImportLookupValue(
        extension, new SkylarkFileDependency(label, fileDependencies.build()));
  }

  /**
   * Converts the PathFragment of the Skylark file to a Label using the BUILD file closest to the
   * Skylark file in its directory hierarchy - finds the package to which the Skylark file belongs.
   * Throws an exception if no such BUILD file exists.
   */
  private Label pathFragmentToLabel(RepositoryName repo, PathFragment file, Environment env)
      throws SkylarkImportLookupFunctionException {
    ContainingPackageLookupValue containingPackageLookupValue = null;
    try {
      PackageIdentifier newPkgId = PackageIdentifier.create(repo, file.getParentDirectory());
      containingPackageLookupValue = (ContainingPackageLookupValue) env.getValueOrThrow(
          ContainingPackageLookupValue.key(newPkgId),
          BuildFileNotFoundException.class, InconsistentFilesystemException.class);
    } catch (BuildFileNotFoundException e) {
      // Thrown when there are IO errors looking for BUILD files.
      throw new SkylarkImportLookupFunctionException(e, Transience.PERSISTENT);
    } catch (InconsistentFilesystemException e) {
      throw new SkylarkImportLookupFunctionException(e, Transience.PERSISTENT);
    }

    if (containingPackageLookupValue == null) {
      return null;
    }

    if (!containingPackageLookupValue.hasContainingPackage()) {
      throw new SkylarkImportLookupFunctionException(
          SkylarkImportFailedException.noBuildFile(file));
    }

    PathFragment pkgName =
        containingPackageLookupValue.getContainingPackageName().getPackageFragment();
    PathFragment fileInPkg = file.relativeTo(pkgName);

    try {
      // This code relies on PackageIdentifier.RepositoryName.toString()
      return Label.parseAbsolute(repo + "//" + pkgName.getPathString() + ":" + fileInPkg);
    } catch (LabelSyntaxException e) {
      throw new IllegalStateException(e);
    }
  }

  /**
   * Creates the Extension to be imported.
   */
  private Extension createExtension(
      BuildFileAST ast,
      PathFragment file,
      Map<PathFragment, Extension> importMap,
      Environment env)
          throws InterruptedException, SkylarkImportLookupFunctionException {
    StoredEventHandler eventHandler = new StoredEventHandler();
    // TODO(bazel-team): this method overestimates the changes which can affect the
    // Skylark RuleClass. For example changes to comments or unused functions can modify the hash.
    // A more accurate - however much more complicated - way would be to calculate a hash based on
    // the transitive closure of the accessible AST nodes.
    try (Mutability mutability = Mutability.create("importing %s", file)) {
      com.google.devtools.build.lib.syntax.Environment extensionEnv =
          ruleClassProvider.createSkylarkRuleClassEnvironment(
              mutability, eventHandler, ast.getContentHashCode(), importMap)
          .setupOverride("native", packageFactory.getNativeModule());
      ast.exec(extensionEnv, eventHandler);
      SkylarkRuleClassFunctions.exportRuleFunctions(extensionEnv, file);

      Event.replayEventsOn(env.getListener(), eventHandler.getEvents());
      if (eventHandler.hasErrors()) {
        throw new SkylarkImportLookupFunctionException(SkylarkImportFailedException.errors(file));
      }
      return new Extension(extensionEnv);
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

    static SkylarkImportFailedException errors(PathFragment file) {
      return new SkylarkImportFailedException(
          String.format("Extension file '%s' has errors", file));
    }

    static SkylarkImportFailedException errorReadingFile(PathFragment file, String error) {
      return new SkylarkImportFailedException(
          String.format("Encountered error while reading extension file '%s': %s", file, error));
    }

    static SkylarkImportFailedException noFile(PathFragment file) {
      return new SkylarkImportFailedException(
          String.format("Extension file not found: '%s'", file));
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
  }

  private static final class SkylarkImportLookupFunctionException extends SkyFunctionException {
    private SkylarkImportLookupFunctionException(SkylarkImportFailedException cause) {
      super(cause, Transience.PERSISTENT);
    }

    private SkylarkImportLookupFunctionException(InconsistentFilesystemException e,
        Transience transience) {
      super(e, transience);
    }

    private SkylarkImportLookupFunctionException(ASTLookupInputException e,
        Transience transience) {
      super(e, transience);
    }

    private SkylarkImportLookupFunctionException(BuildFileNotFoundException e,
        Transience transience) {
      super(e, transience);
    }
  }
}
