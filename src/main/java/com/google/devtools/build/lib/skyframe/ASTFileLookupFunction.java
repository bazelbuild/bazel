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

import com.google.devtools.build.lib.packages.CachingPackageLocator;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicReference;

import javax.annotation.Nullable;

/**
 * A SkyFunction for {@link ASTFileLookupValue}s. Tries to locate a file and load it as a
 * syntax tree and cache the resulting {@link BuildFileAST}. If the file doesn't exist
 * the function doesn't fail but returns a specific NO_FILE ASTLookupValue.
 */
public class ASTFileLookupFunction implements SkyFunction {

  private abstract static class FileLookupResult {
    /** Returns whether the file lookup was successful. */
    public abstract boolean lookupSuccessful();

    /** If {@code lookupSuccessful()}, returns the {@link RootedPath} to the file. */
    public abstract RootedPath rootedPath();

    static FileLookupResult noFile() {
      return UnsuccessfulFileResult.INSTANCE;
    }

    static FileLookupResult file(RootedPath rootedPath) {
      return new SuccessfulFileResult(rootedPath);
    }

    private static class SuccessfulFileResult extends FileLookupResult {
      private final RootedPath rootedPath;

      private SuccessfulFileResult(RootedPath rootedPath) {
        this.rootedPath = rootedPath;
      }

      @Override
      public boolean lookupSuccessful() {
        return true;
      }

      @Override
      public RootedPath rootedPath() {
        return rootedPath;
      }
    }

    private static class UnsuccessfulFileResult extends FileLookupResult {
      private static final UnsuccessfulFileResult INSTANCE = new UnsuccessfulFileResult();
      private UnsuccessfulFileResult() {
      }

      @Override
      public boolean lookupSuccessful() {
        return false;
      }

      @Override
      public RootedPath rootedPath() {
        throw new IllegalStateException("unsucessful lookup");
      }
    }
  }

  private final AtomicReference<PathPackageLocator> pkgLocator;
  private final RuleClassProvider ruleClassProvider;
  private final CachingPackageLocator packageManager;

  public ASTFileLookupFunction(AtomicReference<PathPackageLocator> pkgLocator,
      CachingPackageLocator packageManager,
      RuleClassProvider ruleClassProvider) {
    this.pkgLocator = pkgLocator;
    this.packageManager = packageManager;
    this.ruleClassProvider = ruleClassProvider;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException,
      InterruptedException {
    PathFragment astFilePathFragment = (PathFragment) skyKey.argument();
    FileLookupResult lookupResult = getASTFile(env, astFilePathFragment);
    if (lookupResult == null) {
      return null;
    }

    BuildFileAST ast = null;
    if (!lookupResult.lookupSuccessful()) {
      return ASTFileLookupValue.noFile();
    } else {
      Path path = lookupResult.rootedPath().asPath();
      // Skylark files end with bzl.
      boolean parseAsSkylark = astFilePathFragment.getPathString().endsWith(".bzl");
      try {
        ast = parseAsSkylark
            ? BuildFileAST.parseSkylarkFile(path, env.getListener(),
                packageManager, ruleClassProvider.getSkylarkValidationEnvironment().clone())
            : BuildFileAST.parseBuildFile(path, env.getListener(),
                packageManager, false);
      } catch (IOException e) {
        throw new ASTLookupFunctionException(new ErrorReadingSkylarkExtensionException(
            e.getMessage()), Transience.TRANSIENT);
      }
    }

    return ASTFileLookupValue.withFile(ast);
  }

  private FileLookupResult getASTFile(Environment env, PathFragment astFilePathFragment)
      throws ASTLookupFunctionException {
    for (Path packagePathEntry : pkgLocator.get().getPathEntries()) {
      RootedPath rootedPath = RootedPath.toRootedPath(packagePathEntry, astFilePathFragment);
      SkyKey fileSkyKey = FileValue.key(rootedPath);
      FileValue fileValue = null;
      try {
        fileValue = (FileValue) env.getValueOrThrow(fileSkyKey, IOException.class,
            FileSymlinkException.class, InconsistentFilesystemException.class);
      } catch (IOException | FileSymlinkException e) {
        throw new ASTLookupFunctionException(new ErrorReadingSkylarkExtensionException(
            e.getMessage()), Transience.PERSISTENT);
      } catch (InconsistentFilesystemException e) {
        throw new ASTLookupFunctionException(e, Transience.PERSISTENT);
      }
      if (fileValue == null) {
        return null;
      }
      if (fileValue.isFile()) {
        return FileLookupResult.file(rootedPath);
      }
    }
    return FileLookupResult.noFile();
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
