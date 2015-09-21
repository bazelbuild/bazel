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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.rules.SkylarkRuleClassFunctions;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Environment.Extension;
import com.google.devtools.build.lib.syntax.LoadStatement;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.HashMap;
import java.util.Map;

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
    Label fileLabel = (Label) skyKey.argument();
    PathFragment importPath = fileLabel.toPathFragment();
    ASTFileLookupValue astLookupValue = null;
    try {
      SkyKey astLookupKey = ASTFileLookupValue.key(fileLabel);
      astLookupValue = (ASTFileLookupValue) env.getValueOrThrow(astLookupKey,
          ErrorReadingSkylarkExtensionException.class, InconsistentFilesystemException.class);
    } catch (ErrorReadingSkylarkExtensionException e) {
      throw new SkylarkImportLookupFunctionException(SkylarkImportFailedException.errorReadingFile(
          importPath, e.getMessage()));
    } catch (InconsistentFilesystemException e) {
      throw new SkylarkImportLookupFunctionException(e, Transience.PERSISTENT);
    }
    if (astLookupValue == null) {
      return null;
    }
    if (!astLookupValue.lookupSuccessful()) {
      // Skylark import files have to exist.
      throw new SkylarkImportLookupFunctionException(
          SkylarkImportFailedException.noFile(astLookupValue.getErrorMsg()));
    }

    Map<PathFragment, Extension> importMap = new HashMap<>();
    ImmutableList.Builder<SkylarkFileDependency> fileDependencies = ImmutableList.builder();
    BuildFileAST ast = astLookupValue.getAST();
    for (LoadStatement loadStmt : ast.getImports()) {
      try {
        Label importLabel = findLabelForLoadStatement(loadStmt, fileLabel, env);
        if (importLabel == null) {
          return null;
        }
        SkyKey importsLookupKey = SkylarkImportLookupValue.key(importLabel);
        SkylarkImportLookupValue importsLookupValue;
        importsLookupValue = (SkylarkImportLookupValue) env.getValue(importsLookupKey);
        if (importsLookupValue != null) {
          importMap.put(loadStmt.getImportPath(), importsLookupValue.getEnvironmentExtension());
          fileDependencies.add(importsLookupValue.getDependency());
        }
      } catch (SkylarkImportFailedException e) {
        throw new SkylarkImportLookupFunctionException(e);
      }
    }
    if (env.valuesMissing()) {
      // This means some imports are unavailable.
      return null;
    }

    if (ast.containsErrors()) {
      throw new SkylarkImportLookupFunctionException(
          SkylarkImportFailedException.skylarkErrors(importPath));
    }

    // Skylark UserDefinedFunction-s in that file will share this function definition Environment,
    // which will be frozen by the time it is returned by createExtension.
    Extension extension =
        createExtension(ast, importPath, importMap, env);

    return new SkylarkImportLookupValue(
        extension, new SkylarkFileDependency(fileLabel, fileDependencies.build()));
  }

 /**
  * Given a Skylark {@link LoadStatement} and the {@link Label} of the file containing the load
  * statement, returns a canonical {@link Label} corresponding to the load path in the statement.
  * If no package can be found that contains the loaded file, throws
  * {@link SkylarkImportFailedException}. Returns null if Skyframe dependencies are unavailable.
  */
  @Nullable
  static Label findLabelForLoadStatement(LoadStatement loadStmt, Label containingFileLabel,
      Environment env)
          throws SkylarkImportFailedException {
    PathFragment importPath = loadStmt.getImportPath();
    PackageIdentifier pkgIdForImport;
    String targetNameForImport;
    if (loadStmt.isAbsolute()) {
      PathFragment relativeImportPath = importPath.toRelative();
      PackageIdentifier pkgToLookUp =
          PackageIdentifier.createInDefaultRepo(relativeImportPath.getParentDirectory());
      ContainingPackageLookupValue containingPackageLookupValue = null;
      try {
        containingPackageLookupValue = (ContainingPackageLookupValue) env.getValueOrThrow(
            ContainingPackageLookupValue.key(pkgToLookUp),
            BuildFileNotFoundException.class, InconsistentFilesystemException.class);
      } catch (BuildFileNotFoundException e) {
        // Thrown when there are IO errors looking for BUILD files.
        throw new SkylarkImportFailedException(e);
      } catch (InconsistentFilesystemException e) {
        throw new SkylarkImportFailedException(e);
      }

      if (containingPackageLookupValue == null) {
        return null;
      }

      if (!containingPackageLookupValue.hasContainingPackage()) {
        throw SkylarkImportFailedException.noBuildFile(importPath);
      }
      pkgIdForImport = containingPackageLookupValue.getContainingPackageName();
      PathFragment containingPkgPath = pkgIdForImport.getPackageFragment();  
      targetNameForImport = relativeImportPath.relativeTo(containingPkgPath).toString();
    } else {
      // The load statement has a relative path
      pkgIdForImport = containingFileLabel.getPackageIdentifier();
      PathFragment containingDir =
          (new PathFragment(containingFileLabel.getName())).getParentDirectory();
      targetNameForImport = importPath.getRelative(containingDir).toString();
    }
    try {
      return Label.create(pkgIdForImport, targetNameForImport);
    } catch (LabelSyntaxException e) {
      // Shouldn't happen; the Label is well-formed by construction.
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

    private SkylarkImportFailedException(InconsistentFilesystemException e) {
      super(e.getMessage());
    }

    private SkylarkImportFailedException(BuildFileNotFoundException e) {
      super(e.getMessage());
    }

    static SkylarkImportFailedException errors(PathFragment file) {
      return new SkylarkImportFailedException(
          String.format("Extension file '%s' has errors", file));
    }

    static SkylarkImportFailedException errorReadingFile(PathFragment file, String error) {
      return new SkylarkImportFailedException(
          String.format("Encountered error while reading extension file '%s': %s", file, error));
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
