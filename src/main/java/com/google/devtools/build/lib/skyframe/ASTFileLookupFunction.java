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

import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.InconsistentFilesystemException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkSemantics;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * A SkyFunction for {@link ASTFileLookupValue}s.
 *
 * <p> Given a {@link Label} referencing a Skylark file, loads it as a syntax tree
 * ({@link BuildFileAST}). The Label must be absolute, and must not reference the special
 * {@code external} package. If the file (or the package containing it) doesn't exist, the
 * function doesn't fail, but instead returns a specific {@code NO_FILE} {@link ASTFileLookupValue}.
 */
public class ASTFileLookupFunction implements SkyFunction {

  private final RuleClassProvider ruleClassProvider;

  public ASTFileLookupFunction(RuleClassProvider ruleClassProvider) {
    this.ruleClassProvider = ruleClassProvider;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException,
      InterruptedException {
    Label fileLabel = (Label) skyKey.argument();
    PathFragment filePathFragment = fileLabel.toPathFragment();

    // Determine whether the package designated by fileLabel exists.
    // TODO(bazel-team): After --incompatible_disallow_load_labels_to_cross_package_boundaries is
    // removed and the new behavior is unconditional, we can instead safely assume the package
    // exists and pass in the Root in the SkyKey and therefore this dep can be removed.
    SkyKey pkgSkyKey = PackageLookupValue.key(fileLabel.getPackageIdentifier());
    PackageLookupValue pkgLookupValue = null;
    try {
      pkgLookupValue = (PackageLookupValue) env.getValueOrThrow(
          pkgSkyKey, BuildFileNotFoundException.class, InconsistentFilesystemException.class);
    } catch (BuildFileNotFoundException e) {
      throw new ASTLookupFunctionException(
          new ErrorReadingSkylarkExtensionException(e), Transience.PERSISTENT);
    } catch (InconsistentFilesystemException e) {
      throw new ASTLookupFunctionException(e, Transience.PERSISTENT);
    }
    if (pkgLookupValue == null) {
      return null;
    }
    if (!pkgLookupValue.packageExists()) {
      return ASTFileLookupValue.forBadPackage(fileLabel, pkgLookupValue.getErrorMsg());
    }

    // Determine whether the file designated by fileLabel exists.
    Root packageRoot = pkgLookupValue.getRoot();
    RootedPath rootedPath = RootedPath.toRootedPath(packageRoot, filePathFragment);
    SkyKey fileSkyKey = FileValue.key(rootedPath);
    FileValue fileValue = null;
    try {
      fileValue = (FileValue) env.getValueOrThrow(fileSkyKey, IOException.class);
    } catch (InconsistentFilesystemException e) {
      throw new ASTLookupFunctionException(e, Transience.PERSISTENT);
    } catch (IOException e) {
      throw new ASTLookupFunctionException(
          new ErrorReadingSkylarkExtensionException(e), Transience.PERSISTENT);
    }
    if (fileValue == null) {
      return null;
    }
    if (!fileValue.isFile()) {
      return ASTFileLookupValue.forBadFile(fileLabel);
    }
    SkylarkSemantics skylarkSemantics = PrecomputedValue.SKYLARK_SEMANTICS.get(env);
    if (skylarkSemantics == null) {
      return null;
    }

    // Both the package and the file exist; load the file and parse it as an AST.
    BuildFileAST ast = null;
    Path path = rootedPath.asPath();
    try {
      long astFileSize = fileValue.getSize();
      try (Mutability mutability = Mutability.create("validate")) {
        com.google.devtools.build.lib.syntax.Environment validationEnv =
            ruleClassProvider
                .createSkylarkRuleClassEnvironment(
                    fileLabel,
                    mutability,
                    skylarkSemantics,
                    env.getListener(),
                    // the two below don't matter for extracting the ValidationEnvironment:
                    /*astFileContentHashCode=*/ null,
                    /*importMap=*/ null)
                .setupDynamic(Runtime.PKG_NAME, Runtime.NONE)
                .setupDynamic(Runtime.REPOSITORY_NAME, Runtime.NONE);
        byte[] bytes = FileSystemUtils.readWithKnownFileSize(path, astFileSize);
        ast =
            BuildFileAST.parseSkylarkFile(
                bytes, path.getDigest(), path.asFragment(), env.getListener());
          ast = ast.validate(validationEnv, env.getListener());
        }
    } catch (IOException e) {
      throw new ASTLookupFunctionException(new ErrorReadingSkylarkExtensionException(e),
          Transience.TRANSIENT);
    }

    return ASTFileLookupValue.withFile(ast);
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private static final class ASTLookupFunctionException extends SkyFunctionException {
    private ASTLookupFunctionException(ErrorReadingSkylarkExtensionException e,
        Transience transience) {
      super(e, transience);
    }

    private ASTLookupFunctionException(InconsistentFilesystemException e, Transience transience) {
      super(e, transience);
    }
  }
}
