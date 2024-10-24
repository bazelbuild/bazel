// Copyright 2022 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.Interner;
import com.google.devtools.build.lib.actions.FilesetTraversalParams;
import com.google.devtools.build.lib.actions.FilesetTraversalParams.DirectTraversal;
import com.google.devtools.build.lib.actions.FilesetTraversalParams.DirectTraversalRoot;
import com.google.devtools.build.lib.concurrent.BlazeInterners;

/** A {@link TraversalRequest} for a Fileset, backed by {@link FilesetTraversalParams}. */
public class FilesetTraversalRequest extends TraversalRequest {

  private static final Interner<FilesetTraversalRequest> interner =
      BlazeInterners.newWeakInterner();

  public static FilesetTraversalRequest create(FilesetTraversalParams params) {
    checkNotNull(params.getDirectTraversal(), params);
    return interner.intern(new FilesetTraversalRequest(params));
  }

  private final FilesetTraversalParams params;

  private FilesetTraversalRequest(FilesetTraversalParams params) {
    this.params = params;
  }

  @Override
  public DirectTraversalRoot root() {
    return directTraversal().getRoot();
  }

  @Override
  public final boolean isRootGenerated() {
    return directTraversal().isGenerated();
  }

  @Override
  protected final boolean strictOutputFiles() {
    return directTraversal().isStrictFilesetOutput();
  }

  @Override
  protected boolean skipTestingForSubpackage() {
    return false;
  }

  @Override
  protected boolean emitEmptyDirectoryNodes() {
    return false;
  }

  @Override
  protected final String errorInfo() {
    return String.format(
        "Fileset '%s' traversing file (or directory) '%s'",
        params.getOwnerLabelForErrorMessages(),
        directTraversal().getRoot().getRelativePart().getPathString());
  }

  @Override
  protected final FilesetTraversalRequest duplicateWithOverrides(
      DirectTraversalRoot root, boolean skipTestingForSubpackage) {
    return interner.intern(
        new FilesetTraversalRequestWithOverrides(params, root, skipTestingForSubpackage));
  }

  private DirectTraversal directTraversal() {
    return params.getDirectTraversal();
  }

  @Override
  public final int hashCode() {
    int result = root().hashCode();
    result = 31 * result + Boolean.hashCode(isRootGenerated());
    result = 31 * result + Boolean.hashCode(strictOutputFiles());
    result = 31 * result + Boolean.hashCode(skipTestingForSubpackage());
    return result;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof FilesetTraversalRequest other)) {
      return false;
    }
    return root().equals(other.root())
        && isRootGenerated() == other.isRootGenerated()
        && strictOutputFiles() == other.strictOutputFiles()
        && skipTestingForSubpackage() == other.skipTestingForSubpackage();
  }

  private static final class FilesetTraversalRequestWithOverrides extends FilesetTraversalRequest {
    private final DirectTraversalRoot root;
    private final boolean skipTestingForSubpackage;

    private FilesetTraversalRequestWithOverrides(
        FilesetTraversalParams params, DirectTraversalRoot root, boolean skipTestingForSubpackage) {
      super(params);
      this.root = root;
      this.skipTestingForSubpackage = skipTestingForSubpackage;
    }

    @Override
    public DirectTraversalRoot root() {
      return root;
    }

    @Override
    protected boolean skipTestingForSubpackage() {
      return skipTestingForSubpackage;
    }
  }
}
