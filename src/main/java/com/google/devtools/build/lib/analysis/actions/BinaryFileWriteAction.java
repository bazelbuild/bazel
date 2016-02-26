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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.io.ByteSource;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Preconditions;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Action to write a binary file.
 */
public final class BinaryFileWriteAction extends AbstractFileWriteAction {

  private static final String GUID = "eeee07fe-4b40-11e4-82d6-eba0b4f713e2";

  private final ByteSource source;

  /**
   * Creates a new BinaryFileWriteAction instance without inputs.
   *
   * @param owner the action owner.
   * @param output the Artifact that will be created by executing this Action.
   * @param source a source of bytes that will be written to the file.
   * @param makeExecutable iff true will change the output file to be executable.
   */
  public BinaryFileWriteAction(
      ActionOwner owner, Artifact output, ByteSource source, boolean makeExecutable) {
    super(owner, /*inputs=*/Artifact.NO_ARTIFACTS, output, makeExecutable);
    this.source = Preconditions.checkNotNull(source);
  }

  @VisibleForTesting
  public ByteSource getSource() {
    return source;
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx) {
    return new DeterministicWriter() {
      @Override
      public void writeOutputFile(OutputStream out) throws IOException {
        try (InputStream in = source.openStream()) {
          ByteStreams.copy(in, out);
        }
        out.flush();
      }
    };
  }

  @Override
  protected String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addString(GUID);
    f.addString(String.valueOf(makeExecutable));

    try (InputStream in = source.openStream()) {
      byte[] buffer = new byte[512];
      int amountRead;
      while ((amountRead = in.read(buffer)) != -1) {
        f.addBytes(buffer, 0, amountRead);
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    return f.hexDigestAndReset();
  }
}
