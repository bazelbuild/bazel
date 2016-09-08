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
package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.rules.cpp.FdoSupport.FdoException;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;

import javax.annotation.Nullable;

/**
 * Wrapper for {@link FdoSupport} that turns it into a {@link SkyFunction}.
 */
public class FdoSupportFunction implements SkyFunction {

  /**
   * Wrapper for FDO exceptions.
   */
  public static class FdoSkyException extends SkyFunctionException {
    public FdoSkyException(Exception cause, Transience transience) {
      super(cause, transience);
    }
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws FdoSkyException, InterruptedException {
    BlazeDirectories blazeDirectories = PrecomputedValue.BLAZE_DIRECTORIES.get(env);
    if (env.valuesMissing()) {
      return null;
    }

    // TODO(kchodorow): create a SkyFunction to get the main repository name and pass it in here.
    Path execRoot = blazeDirectories.getExecRoot();
    FdoSupportValue.Key key = (FdoSupportValue.Key) skyKey.argument();
    FdoSupport fdoSupport;
    try {
      fdoSupport = FdoSupport.create(env,
        key.getFdoInstrument(), key.getFdoZip(), key.getLipoMode(), execRoot);
      if (env.valuesMissing()) {
        return null;
      }
    } catch (FdoException e) {
      throw new FdoSkyException(e, Transience.PERSISTENT);
    } catch (IOException e) {
      throw new FdoSkyException(e, Transience.TRANSIENT);
    }

    return new FdoSupportValue(fdoSupport);
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
