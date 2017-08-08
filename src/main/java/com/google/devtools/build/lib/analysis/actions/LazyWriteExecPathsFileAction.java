// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.util.Fingerprint;
import java.io.IOException;
import java.io.OutputStream;

/**
 * Lazily writes the exec path of the given files separated by newline into a specified output file.
 */
public final class LazyWriteExecPathsFileAction extends AbstractFileWriteAction {
  private static final String GUID = "6be94d90-96f3-4bec-8104-1fb08abc2546";

  private final ImmutableSet<Artifact> files;

  public LazyWriteExecPathsFileAction(
      ActionOwner owner, Artifact output, ImmutableSet<Artifact> files) {
    super(owner, Artifact.NO_ARTIFACTS, output, false);
    this.files = files;
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx) {
    return new DeterministicWriter() {
      @Override
      public void writeOutputFile(OutputStream out) throws IOException {
        StringBuilder execPaths = new StringBuilder();
        for (Artifact file : files) {
          if (file.isSourceArtifact()) {
            execPaths.append(file.getExecPathString());
            execPaths.append("\n");
          }
        }
        out.write(execPaths.toString().getBytes(UTF_8));
      }
    };
  }

  /**
   * Computes the Action key for this action by computing the fingerprint for the file contents.
   */
  @Override
  protected String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addString(GUID);
    f.addString(String.valueOf(makeExecutable));
    for (Artifact sourceFile : files) {
      if (sourceFile.isSourceArtifact()) {
        f.addPath(sourceFile.getExecPath());
      }
    }
    return f.hexDigestAndReset();
  }
}