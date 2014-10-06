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
import com.google.devtools.build.lib.skyframe.WorkspaceFileValue.NoSuchBindingException;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.concurrent.atomic.AtomicReference;

/**
 * A class to look up individual bindings defined in the WORKSPACE file.
 *
 * Depends on the WORKSPACE file and the actual target referred to.
 */
public class LabelBindingFunction implements SkyFunction {

  private AtomicReference<PathPackageLocator> pkgLocator;

  public LabelBindingFunction(AtomicReference<PathPackageLocator> pkgLocator) {
    this.pkgLocator = pkgLocator;
  }

  /**
   * Looks up the associated "real" target for the SkyKey's virtual target.
   */
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws LabelBindingFunctionException {
    SkyKey workspaceKey = WorkspaceFileValue.key(pkgLocator.get().getWorkspaceFile());
    WorkspaceFileValue workspaceFile = (WorkspaceFileValue) env.getValue(workspaceKey);
    if (workspaceFile == null) {
      return null;
    }

    Label actualLabel = null;
    try {
      actualLabel = workspaceFile.getActualLabel((Label) skyKey.argument());
    } catch (NoSuchBindingException e) {
      throw new LabelBindingFunctionException(skyKey, e);
    }
    return new LabelBindingValue(actualLabel);
  }

  /**
   * Returns the virtual label.
   */
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private class LabelBindingFunctionException extends SkyFunctionException {
    public LabelBindingFunctionException(SkyKey rootCause, NoSuchBindingException e) {
      super(rootCause, e);
    }
  }
}
