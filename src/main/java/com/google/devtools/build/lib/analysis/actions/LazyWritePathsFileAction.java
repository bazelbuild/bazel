
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
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.util.Fingerprint;
import java.io.IOException;
import java.io.OutputStream;

/**
 * Lazily writes the exec path of the given files separated by newline into a specified output file.
 */
public final class LazyWritePathsFileAction extends AbstractFileWriteAction {
  private static final String GUID = "6be94d90-96f3-4bec-8104-1fb08abc2546";

  private final NestedSet<Artifact> files;
  private final ImmutableSet<Artifact> filesToIgnore;
  private final boolean includeDerivedArtifacts;

  public LazyWritePathsFileAction(
      ActionOwner owner,
      Artifact output,
      NestedSet<Artifact> files,
      ImmutableSet<Artifact> filesToIgnore,
      boolean includeDerivedArtifacts) {
    // TODO(ulfjack): It's a bad idea to have these two constructors do slightly different things.
    super(owner, files, output, false);
    this.files = files;
    this.includeDerivedArtifacts = includeDerivedArtifacts;
    this.filesToIgnore = filesToIgnore;
  }

  public LazyWritePathsFileAction(
      ActionOwner owner,
      Artifact output,
      ImmutableSet<Artifact> files,
      ImmutableSet<Artifact> filesToIgnore,
      boolean includeDerivedArtifacts) {
    super(owner, NestedSetBuilder.emptySet(Order.STABLE_ORDER), output, false);
    this.files = NestedSetBuilder.<Artifact>stableOrder().addAll(files).build();
    this.includeDerivedArtifacts = includeDerivedArtifacts;
    this.filesToIgnore = filesToIgnore;
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx) {
    return new DeterministicWriter() {
      @Override
      public void writeOutputFile(OutputStream out) throws IOException {
        out.write(getContents().toString().getBytes(UTF_8));
      }
    };
  }

  /** Computes the Action key for this action by computing the fingerprint for the file contents. */
  @Override
  protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp) {
    fp.addString(GUID);
    fp.addBoolean(includeDerivedArtifacts);
    fp.addString(getContents());
  }

  private String getContents() {
    StringBuilder stringBuilder = new StringBuilder();
    for (Artifact file : files.toList()) {
      if (filesToIgnore.contains(file)) {
        continue;
      }
      if (file.isSourceArtifact() || includeDerivedArtifacts) {
        stringBuilder.append(file.getRootRelativePathString());
        stringBuilder.append("\n");
      }
    }
    return stringBuilder.toString();
  }
}
