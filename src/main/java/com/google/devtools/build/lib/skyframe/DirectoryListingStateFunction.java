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

import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.FileType;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * A {@link SkyFunction} for {@link DirectoryListingStateValue}s.
 *
 * <p>Merely calls DirectoryListingStateValue#create, but also has special handling for
 * directories outside the package roots (see {@link ExternalFilesHelper}).
 */
public class DirectoryListingStateFunction implements SkyFunction {

  private final ExternalFilesHelper externalFilesHelper;

  /**
   * A file-system abstraction to use. This can e.g. be a {@link DefaultSyscallCache} which helps
   * re-use the results of expensive readdir() operations, that are likely already executed for
   * evaluating globs.
   */
  private final SyscallCache syscallCache;

  public DirectoryListingStateFunction(
      ExternalFilesHelper externalFilesHelper, SyscallCache syscallCache) {
    this.externalFilesHelper = externalFilesHelper;
    this.syscallCache = syscallCache;
  }

  @Override
  @Nullable
  public DirectoryListingStateValue compute(SkyKey skyKey, Environment env)
      throws DirectoryListingStateFunctionException, InterruptedException {
    RootedPath dirRootedPath = (RootedPath) skyKey.argument();

    try {
      FileType fileType = externalFilesHelper.maybeHandleExternalFile(dirRootedPath, env);
      if (env.valuesMissing()) {
        return null;
      }
      return DirectoryListingStateValue.create(
          (fileType.mayBeModifiedByBazel() ? SyscallCache.NO_CACHE : syscallCache)
              .readdir(dirRootedPath.asPath()));
    } catch (ExternalFilesHelper.NonexistentImmutableExternalFileException e) {
      // DirectoryListingStateValue.key assumes the path exists. This exception here is therefore
      // indicative of a programming bug.
      throw new IllegalStateException(dirRootedPath.toString(), e);
    } catch (IOException e) {
      throw new DirectoryListingStateFunctionException(e);
    }
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
