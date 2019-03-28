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
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.UnixGlob.FilesystemCalls;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A {@link SkyFunction} for {@link FileStateValue}s.
 *
 * <p>Merely calls FileStateValue#create, but also has special handling for files outside the
 * package roots (see {@link ExternalFilesHelper}).
 */
public class FileStateFunction implements SkyFunction {

  private final AtomicReference<TimestampGranularityMonitor> tsgm;
  private final AtomicReference<FilesystemCalls> syscallCache;
  private final ExternalFilesHelper externalFilesHelper;

  public FileStateFunction(
      AtomicReference<TimestampGranularityMonitor> tsgm,
      AtomicReference<FilesystemCalls> syscallCache,
      ExternalFilesHelper externalFilesHelper) {
    this.tsgm = tsgm;
    this.syscallCache = syscallCache;
    this.externalFilesHelper = externalFilesHelper;
  }

  @Override
  public FileStateValue compute(SkyKey skyKey, Environment env)
      throws FileStateFunctionException, InterruptedException {
    RootedPath rootedPath = (RootedPath) skyKey.argument();

    try {
      externalFilesHelper.maybeHandleExternalFile(rootedPath, false, env);
      if (env.valuesMissing()) {
        return null;
      }
      return FileStateValue.create(rootedPath, syscallCache.get(), tsgm.get());
    } catch (ExternalFilesHelper.NonexistentImmutableExternalFileException e) {
      return FileStateValue.NONEXISTENT_FILE_STATE_NODE;
    } catch (IOException e) {
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
    FileStateFunctionException(IOException e) {
      super(e, Transience.TRANSIENT);
    }
  }
}
