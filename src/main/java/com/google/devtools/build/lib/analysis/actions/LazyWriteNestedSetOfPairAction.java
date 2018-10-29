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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Pair;
import java.io.IOException;
import java.io.OutputStream;

/**
 * Lazily writes the content of a nested set of pairsToWrite to an output file.
 *
 * <p>For each pair <string1, string2> it writes a line string1:string2 to the output file.
 */
public final class LazyWriteNestedSetOfPairAction extends AbstractFileWriteAction {
  private static final String GUID = "cd68d80a-16fc-41dc-a3dd-ded5aa5f601f";

  private final NestedSet<Pair<String, String>> pairsToWrite;
  private String fileContents;

  public LazyWriteNestedSetOfPairAction(
      ActionOwner owner, Artifact output, NestedSet<Pair<String, String>> pairsToWrite) {
    super(owner, ImmutableList.of(), output, /* makeExecutable= */ false);
    this.pairsToWrite = pairsToWrite;
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx) {
    return new DeterministicWriter() {
      @Override
      public void writeOutputFile(OutputStream out) throws IOException {
        out.write(getContents().getBytes(UTF_8));
      }
    };
  }

  /** Computes the Action key for this action by computing the fingerprint for the file contents. */
  @Override
  protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp) {
    actionKeyContext.addNestedSetToFingerprint(fp, pairsToWrite);
  }

  private String getContents() {
    if (fileContents == null) {
      StringBuilder stringBuilder = new StringBuilder();
      for (Pair<String, String> pair : pairsToWrite) {
        stringBuilder
            .append(pair.first)
            .append(":")
            .append(pair.second)
            .append(System.lineSeparator());
      }
      fileContents = stringBuilder.toString();
    }
    return fileContents;
  }
}
