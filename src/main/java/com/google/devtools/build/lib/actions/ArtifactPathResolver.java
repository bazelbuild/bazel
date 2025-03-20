// Copyright 2018 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import javax.annotation.Nullable;

/**
 * An indirection layer on Path resolution of {@link Artifact} and {@link Root}.
 *
 * <p>Serves as converter interface primarily for switching the {@link FileSystem} underlying the
 * values.
 */
public interface ArtifactPathResolver {

  /**
   * @return a resolved Path corresponding to the given actionInput.
   */
  Path toPath(ActionInput actionInput);

  /**
   * @return a resolved Path corresponding to the given path.
   */
  Path convertPath(Path path);

  /** @return a resolved {@link Root} corresponding to the given Root. */
  Root transformRoot(Root root);

  ArtifactPathResolver IDENTITY = new IdentityResolver(null);

  static ArtifactPathResolver forExecRoot(Path execRoot) {
    return new IdentityResolver(execRoot);
  }

  static ArtifactPathResolver withTransformedFileSystem(Path execRoot) {
    return new TransformResolver(execRoot);
  }

  static ArtifactPathResolver createPathResolver(@Nullable FileSystem fileSystem,
      Path execRoot) {
    if (fileSystem == null) {
      return forExecRoot(execRoot);
    } else {
      return withTransformedFileSystem(
          fileSystem.getPath(execRoot.asFragment()));
    }
  }

  /**
   * Path resolution that uses an Artifact's path directly, or looks up the input execPath relative
   * to the given execRoot.
   */
  class IdentityResolver implements ArtifactPathResolver {
    private final Path execRoot;

    private IdentityResolver(Path execRoot) {
      this.execRoot = execRoot;
    }

    @Override
    public Path toPath(ActionInput actionInput) {
      if (actionInput instanceof Artifact artifact) {
        return artifact.getPath();
      }
      return execRoot.getRelative(actionInput.getExecPath());
    }

    @Override
    public Root transformRoot(Root root) {
      return Preconditions.checkNotNull(root);
    }

    @Override
    public Path convertPath(Path path) {
      return path;
    }
  };

  /**
   * A resolver that transforms all results to the same filesystem as the given execRoot.
   */
  class TransformResolver implements ArtifactPathResolver {
    private final FileSystem fileSystem;
    private final Path execRoot;

    private TransformResolver(Path execRoot) {
      this.execRoot = execRoot;
      this.fileSystem = Preconditions.checkNotNull(execRoot.getFileSystem());
    }

    @Override
    public Path toPath(ActionInput input) {
      if (input instanceof Artifact artifact) {
        return fileSystem.getPath(artifact.getPath().asFragment());
      }
      return execRoot.getRelative(input.getExecPath());
    }

    @Override
    public Root transformRoot(Root root) {
      return Root.toFileSystem(Preconditions.checkNotNull(root), fileSystem);
    }

    @Override
    public Path convertPath(Path path) {
      return fileSystem.getPath(path.asFragment());
    }
  }
}
