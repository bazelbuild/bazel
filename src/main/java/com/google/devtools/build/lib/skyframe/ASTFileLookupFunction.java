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
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicReference;

import javax.annotation.Nullable;

/**
 * A SkyFunction for {@link ASTLookupValue}s. Tries to locate a file and load it as a
 * syntax tree and cache the resulting {@link Statement}s.
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

  public ASTFileLookupFunction(AtomicReference<PathPackageLocator> pkgLocator) {
    this.pkgLocator = pkgLocator;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException,
      InterruptedException {
    PathFragment packagePathFragment = (PathFragment) skyKey.argument();

    FileLookupResult lookup = getASTFile(skyKey, env, packagePathFragment);
    if (lookup == null) {
      return null;
    }

    ImmutableList<Statement> preludeStatements = ImmutableList.<Statement>of();
    // TODO(bazel-team): We silently ignore non existent file cases and return an empty list of
    // Statements. This behaviour should be optional in the future if we use this class to
    // look up BUILD files.
    if (lookup.result != FileLookupResultState.NO_FILE) {
      Path path = lookup.file.realRootedPath().asPath();
      try {
        preludeStatements = ImmutableList.copyOf(BuildFileAST.parseBuildFile(
            path, env.getListener(), null, false).getStatements());
      } catch (IOException e) {
        throw new ASTLookupFunctionException(skyKey, e);
      }
    }

    return new ASTLookupValue(preludeStatements);
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
        throw new ASTLookupFunctionException(skyKey, e);
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
    private ASTLookupFunctionException(SkyKey key, IOException e) {
      super(key, e);
    }
  }
}
