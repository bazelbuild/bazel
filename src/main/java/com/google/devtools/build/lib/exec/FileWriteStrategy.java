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

package com.google.devtools.build.lib.exec;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.view.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.view.actions.FileWriteActionContext;
import com.google.devtools.build.lib.view.fileset.FilesetManifestAction;
import com.google.devtools.common.options.OptionsClassProvider;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;

/**
 * A strategy for executing an {@link AbstractFileWriteAction}.
 */
@ExecutionStrategy(contextType = FileWriteActionContext.class)
public final class FileWriteStrategy implements FileWriteActionContext {

  public static final Class<FileWriteStrategy> TYPE = FileWriteStrategy.class;

  // Can consume a lot of CPU, but several concurrent FilesetManifestAction instances will play
  // nicely together, as they consume a shared thread pool.
  private static final ResourceSet FILESET_MANIFEST_ACTION_RESOURCE_SET =
      new ResourceSet(/*memoryMb=*/100, /*cpuUsage=*/.2, /*ioUsage=*/0.0);

  // We're mainly doing I/O, so estimate very low CPU usage, e.g. 1%. Just a guess.
  private static final ResourceSet DEFAULT_FILEWRITE_ACTION_RESOURCE_SET =
      new ResourceSet(/*memoryMb=*/0.0, /*cpuUsage=*/0.01, /*ioUsage=*/0.0);

  public FileWriteStrategy(OptionsClassProvider options) {
  }

  @Override
  public void exec(Executor executor, AbstractFileWriteAction action,
      FileOutErr outErr) throws ExecException, InterruptedException {
    EventHandler reporter = executor == null ? null : executor.getEventHandler();
    try {
      Path outputPath = Iterables.getOnlyElement(action.getOutputs()).getPath();
      OutputStream out = new BufferedOutputStream(outputPath.getOutputStream());
      try {
        action.writeOutputFile(out, reporter, executor);
      } finally {
        out.close();
      }
      if (action.makeExecutable()) {
        outputPath.setExecutable(true);
      }
    } catch (IOException e) {
      throw new EnvironmentalExecException("failed to create file '"
          + Iterables.getOnlyElement(action.getOutputs()).prettyPrint()
          + "' due to I/O error: " + e.getMessage(), e);
    }
  }

  @Override
  public ResourceSet estimateResourceConsumption(AbstractFileWriteAction action) {
    if (action instanceof FilesetManifestAction) {
      return FILESET_MANIFEST_ACTION_RESOURCE_SET;
    } else {
      return DEFAULT_FILEWRITE_ACTION_RESOURCE_SET;
    }
  }
}
