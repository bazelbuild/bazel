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

package com.google.devtools.build.lib.actions;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.vfs.DigestUtils;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.XattrProvider;
import java.io.IOException;
import javax.annotation.Nullable;

/** Helper utility to create ActionInput instances. */
public final class ActionInputHelper {
  private ActionInputHelper() {}

  /**
   * Most ActionInputs are created and never used again. On the off chance that one is, however, we
   * implement equality via path comparison. Since file caches are keyed by ActionInput, equality
   * checking does come up.
   */
  public abstract static class BasicActionInput implements ActionInput {

    // TODO(lberki): Plumb this flag from InputTree.build() somehow.
    @Override
    public boolean isSymlink() {
      return false;
    }

    @Override
    public boolean isDirectory() {
      return false;
    }

    @Override
    public int hashCode() {
      return getExecPathString().hashCode();
    }

    @Override
    public boolean equals(Object other) {
      if (this == other) {
        return true;
      }
      if (!(other instanceof BasicActionInput)) {
        return false;
      }
      return getExecPathString().equals(((BasicActionInput) other).getExecPathString());
    }

    @Override
    public String toString() {
      return "BasicActionInput: " + getExecPathString();
    }
  }

  /**
   * Creates an ActionInput with just the given relative path and no digest.
   *
   * @param path the relative path of the input.
   * @return a ActionInput.
   */
  public static ActionInput fromPath(String path) {
    return new BasicActionInput() {
      @Override
      public String getExecPathString() {
        return path;
      }

      @Override
      public PathFragment getExecPath() {
        return PathFragment.create(path);
      }
    };
  }

  /**
   * Creates an ActionInput with just the given relative path and no digest.
   *
   * @param path the relative path of the input.
   * @return a ActionInput.
   */
  public static ActionInput fromPath(PathFragment path) {
    return new BasicActionInput() {
      @Override
      public String getExecPathString() {
        return path.getPathString();
      }

      @Override
      public PathFragment getExecPath() {
        return path;
      }
    };
  }

  public static Iterable<String> toExecPaths(Iterable<? extends ActionInput> artifacts) {
    return Iterables.transform(artifacts, ActionInput::getExecPathString);
  }

  /** Returns the {@link Path} for an {@link ActionInput}. */
  public static Path toInputPath(ActionInput input, Path execRoot) {
    Preconditions.checkNotNull(input, "input");
    Preconditions.checkNotNull(execRoot, "execRoot");

    return input instanceof Artifact artifact
        ? artifact.getPath()
        : execRoot.getRelative(input.getExecPath());
  }

  /**
   * Returns the digest of the file of an action input, read at {@code path}, using the digest cache
   * under the input's exec path. See {@link DigestUtils#getDigestWithManualFallback(PathFragment,
   * Path, XattrProvider, FileStatus)}.
   *
   * @param path the path to stat and read, which may differ from the input's own path, such as a
   *     path on an action filesystem, but always ends with the input's exec path
   */
  public static byte[] getDigestWithManualFallback(
      ActionInput input, Path path, XattrProvider xattrProvider, @Nullable FileStatus status)
      throws IOException {
    return DigestUtils.getDigestWithManualFallback(
        input.getExecPath(), path, xattrProvider, status);
  }

  /**
   * Same as {@link #getDigestWithManualFallback(ActionInput, Path, XattrProvider, FileStatus)}
   * when a fast digest is known to be unavailable. See {@link
   * DigestUtils#manuallyComputeDigest(PathFragment, Path, FileStatus)}.
   */
  public static byte[] manuallyComputeDigest(
      ActionInput input, Path path, @Nullable FileStatus status) throws IOException {
    return DigestUtils.manuallyComputeDigest(input.getExecPath(), path, status);
  }
}
