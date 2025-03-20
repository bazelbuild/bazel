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

import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.io.InconsistentFilesystemException;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.FileType;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * A {@link SkyFunction} for {@link FileStateValue}s.
 *
 * <p>Merely calls FileStateValue#create, but also has special handling for files outside the
 * package roots (see {@link ExternalFilesHelper}).
 */
public class FileStateFunction implements SkyFunction {

  private final Supplier<TimestampGranularityMonitor> tsgm;
  private final SyscallCache syscallCache;
  private final ExternalFilesHelper externalFilesHelper;

  public FileStateFunction(
      Supplier<TimestampGranularityMonitor> tsgm,
      SyscallCache syscallCache,
      ExternalFilesHelper externalFilesHelper) {
    this.tsgm = tsgm;
    this.syscallCache = syscallCache;
    this.externalFilesHelper = externalFilesHelper;
  }

  // InconsistentFilesystemException catch block needs to be separate from IOException catch block
  // below because Java does "single dispatch": the runtime type of e is all that is considered when
  // deciding which overload of FileStateFunctionException() to call.
  @SuppressWarnings("UseMultiCatch")
  @Override
  @Nullable
  public FileStateValue compute(SkyKey skyKey, Environment env)
      throws FileStateFunctionException, InterruptedException {
    RootedPath rootedPath = (RootedPath) skyKey.argument();

    try {
      FileType fileType = externalFilesHelper.maybeHandleExternalFile(rootedPath, env);
      if (env.valuesMissing()) {
        return null;
      }
      if (fileType == FileType.EXTERNAL_REPO) {
        // do not use syscallCache as files under repositories get generated during the build
        return FileStateValue.create(rootedPath, SyscallCache.NO_CACHE, tsgm.get());
      }
      return FileStateValue.create(rootedPath, syscallCache, tsgm.get());
    } catch (ExternalFilesHelper.NonexistentImmutableExternalFileException e) {
      return FileStateValue.NONEXISTENT_FILE_STATE_NODE;
    } catch (InconsistentFilesystemException e) {
      throw new FileStateFunctionException(e);
    } catch (IOException e) {
      throw new FileStateFunctionException(e);
    }
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by {@link
   * FileStateFunction#compute}.
   */
  public static final class FileStateFunctionException extends SkyFunctionException {
    private final boolean isCatastrophic;

    private FileStateFunctionException(InconsistentFilesystemException e) {
      super(e, Transience.TRANSIENT);
      this.isCatastrophic = true;
    }

    private FileStateFunctionException(IOException e) {
      super(e, Transience.TRANSIENT);
      this.isCatastrophic = false;
    }

    @Override
    public boolean isCatastrophic() {
      return isCatastrophic;
    }
  }
}
