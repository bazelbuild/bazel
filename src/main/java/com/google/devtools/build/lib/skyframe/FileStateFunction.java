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
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A {@link SkyFunction} for {@link FileStateValue}s.
 *
 * <p>Merely calls FileStateValue#create, but also adds a dep on the build id for files outside
 * the package roots.
 */
public class FileStateFunction implements SkyFunction {

  private final TimestampGranularityMonitor tsgm;
  private final AtomicReference<PathPackageLocator> pkgLocator;

  public FileStateFunction(TimestampGranularityMonitor tsgm,
      AtomicReference<PathPackageLocator> pkgLocator) {
    this.tsgm = tsgm;
    this.pkgLocator = pkgLocator;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws FileStateFunctionException {
    RootedPath rootedPath = (RootedPath) skyKey.argument();
    if (!pkgLocator.get().getPathEntries().contains(rootedPath.getRoot())) {
      // For files outside the package roots, add a dependency on the build_id. This is sufficient
      // for correctness; all other files will be handled by diff awareness of their respective
      // package path, but these files need to be addressed separately.
      //
      // Using the build_id here seems to introduce a performance concern because the upward
      // transitive closure of these external files will get eagerly invalidated on each
      // incremental build (e.g. if every file had a transitive dependency on the filesystem root,
      // then we'd have a big performance problem). But this a non-issue by design:
      // - We don't add a dependency on the parent directory at the package root boundary, so the
      // only transitive dependencies from files inside the package roots to external files are
      // through symlinks. So the upwards transitive closure of external files is small.
      // - The only way external source files get into the skyframe graph in the first place is
      // through symlinks outside the package roots, which we neither want to encourage nor
      // optimize for since it is not common. So the set of external files is small.
      UUID buildId = PrecomputedValue.BUILD_ID.get(env);
      if (buildId == null) {
        return null;
      }
    }
    try {
      return FileStateValue.create(rootedPath, tsgm);
    } catch (IOException e) {
      throw new FileStateFunctionException(e);
    } catch (InconsistentFilesystemException e) {
      throw new FileStateFunctionException(e);
    }
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link FileStateFunction#compute}.
   */
  private static final class FileStateFunctionException extends SkyFunctionException {
    public FileStateFunctionException(IOException e) {
      super(e, Transience.TRANSIENT);
    }

    public FileStateFunctionException(InconsistentFilesystemException e) {
      super(e, Transience.TRANSIENT);
    }
  }
}
