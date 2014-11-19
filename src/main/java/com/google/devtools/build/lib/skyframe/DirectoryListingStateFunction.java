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

import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A {@link SkyFunction} for {@link DirectoryListingStateValue}s.
 *
 * <p>Merely calls DirectoryListingStateValue#create, but also adds a dep on the build id for
 * directories outside the package roots.
 */
public class DirectoryListingStateFunction implements SkyFunction {

  private final AtomicReference<PathPackageLocator> pkgLocator;

  public DirectoryListingStateFunction(AtomicReference<PathPackageLocator> pkgLocator) {
    this.pkgLocator = pkgLocator;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws DirectoryListingStateFunctionException {
    RootedPath dirRootedPath = (RootedPath) skyKey.argument();
    // See the note about external files in FileStateFunction.
    if (!pkgLocator.get().getPathEntries().contains(dirRootedPath.getRoot())) {
      UUID buildId = PrecomputedValue.BUILD_ID.get(env);
      if (buildId == null) {
        return null;
      }
    }
    try {
      return DirectoryListingStateValue.create(dirRootedPath);
    } catch (IOException e) {
      throw new DirectoryListingStateFunctionException(e);
    }
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link DirectoryListingStateFunction#compute}.
   */
  private static final class DirectoryListingStateFunctionException
      extends SkyFunctionException {
    public DirectoryListingStateFunctionException(IOException e) {
      super(e, Transience.TRANSIENT);
    }
  }
}
