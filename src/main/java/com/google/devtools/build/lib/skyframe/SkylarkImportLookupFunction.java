// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageIdentifier;
import com.google.devtools.build.lib.packages.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.SkylarkNativeModule;
import com.google.devtools.build.lib.skyframe.ASTFileLookupValue.ASTLookupInputException;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Function;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.Label.SyntaxException;
import com.google.devtools.build.lib.syntax.SkylarkEnvironment;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.HashMap;
import java.util.Map;

/**
 * A Skyframe function to look up and import a single Skylark extension.
 */
public class SkylarkImportLookupFunction implements SkyFunction {

  private final RuleClassProvider ruleClassProvider;
  private final ImmutableList<Function> nativeRuleFunctions;

  public SkylarkImportLookupFunction(
      RuleClassProvider ruleClassProvider, PackageFactory packageFactory) {
    this.ruleClassProvider = ruleClassProvider;
    this.nativeRuleFunctions = packageFactory.collectNativeRuleFunctions();
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException,
      InterruptedException {
    PackageIdentifier arg = (PackageIdentifier) skyKey.argument();
    PathFragment file = arg.getPackageFragment();
    ASTFileLookupValue astLookupValue = null;
    try {
      SkyKey astLookupKey = ASTFileLookupValue.key(file);
      astLookupValue = (ASTFileLookupValue) env.getValueOrThrow(astLookupKey,
          ErrorReadingSkylarkExtensionException.class, InconsistentFilesystemException.class);
    } catch (ErrorReadingSkylarkExtensionException e) {
      throw new SkylarkImportLookupFunctionException(SkylarkImportFailedException.errorReadingFile(
          file, e.getMessage()));
    } catch (InconsistentFilesystemException e) {
      throw new SkylarkImportLookupFunctionException(e, Transience.PERSISTENT);
    } catch (ASTLookupInputException e) {
      throw new SkylarkImportLookupFunctionException(e, Transience.PERSISTENT);
    }
    if (astLookupValue == null) {
      return null;
    }
    if (astLookupValue.getAST() == null) {
      // Skylark import files have to exist.
      throw new SkylarkImportLookupFunctionException(SkylarkImportFailedException.noFile(file));
    }

    Map<PathFragment, SkylarkEnvironment> importMap = new HashMap<>();
    ImmutableList.Builder<SkylarkFileDependency> fileDependencies = ImmutableList.builder();
    BuildFileAST ast = astLookupValue.getAST();
    // TODO(bazel-team): Refactor this code and PackageFunction to reduce code duplications.
    for (PathFragment importFile : ast.getImports()) {
      try {
        SkyKey importsLookupKey =
            SkylarkImportLookupValue.key(arg.getRepository(), file, importFile);
        SkylarkImportLookupValue importsLookupValue;
        importsLookupValue = (SkylarkImportLookupValue) env.getValueOrThrow(
            importsLookupKey, ASTLookupInputException.class);
        if (importsLookupValue != null) {
          importMap.put(importFile, importsLookupValue.getImportedEnvironment());
          fileDependencies.add(importsLookupValue.getDependency());
        }
      } catch (ASTLookupInputException e) {
        throw new SkylarkImportLookupFunctionException(e, Transience.PERSISTENT);
      }
    }
    Label label = pathFragmentToLabel(arg.getRepository(), file, env);
    if (env.valuesMissing()) {
      // This means some imports are unavailable.
      return null;
    }

    if (ast.containsErrors()) {
      throw new SkylarkImportLookupFunctionException(SkylarkImportFailedException.skylarkErrors(
          file));
    }

    SkylarkEnvironment extensionEnv = createEnv(ast, file, importMap, env);
    // Skylark UserDefinedFunctions are sharing function definition Environments, so it's extremely
    // important not to modify them from this point. Ideally they should be only used to import
    // symbols and serve as global Environments of UserDefinedFunctions.
    return new SkylarkImportLookupValue(
        extensionEnv, new SkylarkFileDependency(label, fileDependencies.build()));
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
      PackageIdentifier newPkgId = new PackageIdentifier(repo, file.getParentDirectory());
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
      return Label.parseRepositoryLabel(repo + "//" + pkgName.getPathString() + ":" + fileInPkg);
    } catch (SyntaxException e) {
      throw new IllegalStateException(e);
    }
  }

  /**
   * Creates the SkylarkEnvironment to be imported. After it's returned, the Environment
   * must not be modified.
   */
  private SkylarkEnvironment createEnv(BuildFileAST ast, PathFragment file,
      Map<PathFragment, SkylarkEnvironment> importMap, Environment env)
          throws InterruptedException, SkylarkImportLookupFunctionException {
    StoredEventHandler eventHandler = new StoredEventHandler();
    // TODO(bazel-team): this method overestimates the changes which can affect the
    // Skylark RuleClass. For example changes to comments or unused functions can modify the hash.
    // A more accurate - however much more complicated - way would be to calculate a hash based on
    // the transitive closure of the accessible AST nodes.
    SkylarkEnvironment extensionEnv = ruleClassProvider
        .createSkylarkRuleClassEnvironment(eventHandler, ast.getContentHashCode());
    // Adding native rules module for build extensions.
    // TODO(bazel-team): this might not be the best place to do this.
    for (Function function : nativeRuleFunctions) {
        extensionEnv.registerFunction(
            SkylarkNativeModule.class, function.getName(), function);
    }
    extensionEnv.setImportedExtensions(importMap);
    ast.exec(extensionEnv, eventHandler);

    Event.replayEventsOn(env.getListener(), eventHandler.getEvents());
    if (eventHandler.hasErrors()) {
      throw new SkylarkImportLookupFunctionException(SkylarkImportFailedException.errors(file));
    }
    return extensionEnv;
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
