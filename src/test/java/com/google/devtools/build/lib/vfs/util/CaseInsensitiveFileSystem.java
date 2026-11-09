// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs.util;

import com.google.common.base.Ascii;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.PathTransformingDelegateFileSystem;
import java.io.IOException;

/**
 * A case-preserving file system that resolves each path segment to an existing entry that only
 * differs in ASCII case, like the default file systems on macOS and Windows.
 */
public final class CaseInsensitiveFileSystem extends PathTransformingDelegateFileSystem {

  public CaseInsensitiveFileSystem(FileSystem delegateFs) {
    super(delegateFs);
  }

  @Override
  public boolean mayBeCaseOrNormalizationInsensitive() {
    return true;
  }

  @Override
  protected PathFragment toDelegatePath(PathFragment path) {
    PathFragment resolved = path.subFragment(0, 0);
    for (String segment : path.segments()) {
      resolved = resolveChild(resolved, segment);
    }
    return resolved;
  }

  @Override
  protected PathFragment fromDelegatePath(PathFragment delegatePath) {
    return delegatePath;
  }

  @Override
  public void renameTo(PathFragment sourcePath, PathFragment targetPath) throws IOException {
    // Keep the spelling of the target's base name so that renames can change the case of an entry.
    PathFragment targetParent = toDelegatePath(targetPath.getParentDirectory());
    PathFragment resolvedTarget = resolveChild(targetParent, targetPath.getBaseName());
    PathFragment resolvedSource = toDelegatePath(sourcePath);
    getDelegateFs()
        .renameTo(
            resolvedSource,
            resolvedTarget.equals(resolvedSource)
                ? targetParent.getChild(targetPath.getBaseName())
                : resolvedTarget);
  }

  private PathFragment resolveChild(PathFragment resolvedParent, String name) {
    PathFragment exact = resolvedParent.getChild(name);
    try {
      if (getDelegateFs().exists(exact, /* followSymlinks= */ false)) {
        return exact;
      }
      for (String entry : getDelegateFs().getDirectoryEntries(resolvedParent)) {
        if (Ascii.equalsIgnoreCase(entry, name)) {
          return resolvedParent.getChild(entry);
        }
      }
    } catch (IOException e) {
      // The parent doesn't exist or isn't a directory, so neither does the child.
    }
    return exact;
  }
}
