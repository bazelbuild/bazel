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

package com.google.devtools.build.lib.analysis.actions;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;

/**
 * Action to create an executable symbolic link. It includes additional validation that symlink
 * target is indeed an executable file.
 */
@AutoCodec
public final class ExecutableSymlinkAction extends SymlinkAction {

  @AutoCodec.Instantiator
  public ExecutableSymlinkAction(ActionOwner owner, Artifact primaryInput, Artifact primaryOutput) {
    super(owner, null, primaryInput, primaryOutput, "Symlinking " + owner.getLabel(), false);
  }

  @Override
  public ActionResult execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException {
    Path inputPath = actionExecutionContext.getInputPath(getPrimaryInput());
    try {
      // Validate that input path is a file with the executable bit is set.
      if (!inputPath.isFile()) {
        throw new ActionExecutionException(
            "'" + Iterables.getOnlyElement(getInputs()).prettyPrint() + "' is not a file", this,
            false);
      }
      if (!inputPath.isExecutable()) {
        throw new ActionExecutionException("failed to create symbolic link '"
            + Iterables.getOnlyElement(getOutputs()).prettyPrint()
            + "': file '" + Iterables.getOnlyElement(getInputs()).prettyPrint()
            + "' is not executable", this, false);
      }
    } catch (IOException e) {
      throw new ActionExecutionException("failed to create symbolic link '"
          + Iterables.getOnlyElement(getOutputs()).prettyPrint()
          + "' to the '" + Iterables.getOnlyElement(getInputs()).prettyPrint()
          + "' due to I/O error: " + e.getMessage(), e, this, false);
    }

    return super.execute(actionExecutionContext);
  }

  @Override
  public String getMnemonic() {
    return "ExecutableSymlink";
  }
}
