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
 * A SkyFunction for {@link ASTLookupValue}s. Tries to locate a file and load it as a
 * syntax tree and cache the resulting {@link BuildFileAST}. If the file doesn't exist
 * the function doesn't fail but returns a specific NO_FILE ASTLookupValue.
 */
public class ASTFileLookupFunction implements SkyFunction {

  private static enum FileLookupResultState {
    SUCCESS,
    NO_FILE;
  }

  private static final class FileLookupResult {
    private final FileValue file;
    private final FileLookupResultState result;

    private FileLookupResult(FileLookupResultState result, FileValue file) {
      this.file = file;
      this.result = result;
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
    PathFragment packagePathFragment = (PathFragment) skyKey.argument();

    FileLookupResult lookup = getASTFile(skyKey, env, packagePathFragment);
    if (lookup == null) {
      return null;
    }

    BuildFileAST ast = null;
    if (lookup.result == FileLookupResultState.NO_FILE) {
      // Return the specific NO_FILE ASTLookupValue instance if no file was found.
      return ASTLookupValue.NO_FILE;
    } else {
      Path path = lookup.file.realRootedPath().asPath();
      try {
        ast = BuildFileAST.parseSkylarkFile(path, env.getListener(),
            packageManager, ruleClassProvider.getSkylarkValidationEnvironment().clone());
      } catch (IOException e) {
        throw new ASTLookupFunctionException(skyKey, e, Transience.TRANSIENT);
      }
    }

    return new ASTLookupValue(ast);
  }

  private FileLookupResult getASTFile(SkyKey skyKey, Environment env,
      PathFragment packagePathFragment) throws ASTLookupFunctionException {
    for (Path packagePathEntry : pkgLocator.get().getPathEntries()) {
      RootedPath rootedPath = RootedPath.toRootedPath(packagePathEntry, packagePathFragment);
      SkyKey fileSkyKey = FileValue.key(rootedPath);
      FileValue fileValue = null;
      try {
        fileValue = (FileValue) env.getValueOrThrow(fileSkyKey, Exception.class);
      } catch (IOException e) {
        throw new ASTLookupFunctionException(skyKey, e, Transience.PERSISTENT);
      } catch (Exception e) {
        throw new IllegalStateException("Exception when loading AST file", e);
      }
      if (fileValue == null) {
        return null;
      }
      if (fileValue.isFile()) {
        return new FileLookupResult(FileLookupResultState.SUCCESS, fileValue);
      }
    }
    return new FileLookupResult(FileLookupResultState.NO_FILE, null);
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private static final class ASTLookupFunctionException extends SkyFunctionException {
    private ASTLookupFunctionException(SkyKey key, IOException e, Transience transience) {
      super(key, e, transience);
    }
  }
}
