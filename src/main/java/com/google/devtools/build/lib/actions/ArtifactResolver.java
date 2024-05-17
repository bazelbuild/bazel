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

import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * An interface for resolving artifact names to {@link Artifact} objects. Should only be used
 * in the internal machinery of Blaze: rule implementations are not allowed to do this.
 */
public interface ArtifactResolver {
  /**
   * Returns the {@link SourceArtifact} for the specified path, creating it if not found and setting
   * its root and execPath.
   *
   * @param execPath the path of the source artifact relative to the source root
   * @param root the source root prefix of the path
   * @param owner the artifact owner.
   * @return the canonical source artifact for the given path
   */
  SourceArtifact getSourceArtifact(PathFragment execPath, Root root, ArtifactOwner owner);

  /**
   * Returns the {@link SourceArtifact} for the specified path, creating it if not found and setting
   * its root and execPath.
   *
   * @see #getSourceArtifact(PathFragment, Root, ArtifactOwner)
   */
  SourceArtifact getSourceArtifact(PathFragment execPath, Root root);

  /**
   * Resolves a {@link SourceArtifact} given an execRoot-relative path.
   *
   * <p>Note: this method should only be used when the roots are unknowable, such as from the
   * post-compile .d or manifest scanning methods.
   *
   * @param execPath the exec path of the artifact to resolve
   * @param repositoryName the name of repository this artifact belongs to
   * @return an existing or new source Artifact for the given execPath. Returns null if the root can
   *     not be determined and the artifact did not exist before.
   */
  SourceArtifact resolveSourceArtifact(PathFragment execPath, RepositoryName repositoryName);

  /**
   * Resolves source Artifacts given execRoot-relative paths.
   *
   * <p>Never creates or returns derived artifacts, only source artifacts.
   *
   * <p>Note: this method should only be used when the roots are unknowable, such as from the
   * post-compile .d or manifest scanning methods.
   *
   * @param execPaths list of exec paths of the artifacts to resolve
   * @param resolver object that helps to resolve package root of given paths
   * @return map which contains list of execPaths and corresponding Artifacts. Map can contain
   *     existing or new source Artifacts for the given execPaths. The artifact is null if the root
   *     cannot be determined and the artifact did not exist before. Return null if any dependencies
   *     are missing.
   */
  @Nullable
  Map<PathFragment, SourceArtifact> resolveSourceArtifacts(
      Iterable<PathFragment> execPaths, PackageRootResolver resolver)
      throws PackageRootResolver.PackageRootException, InterruptedException;

  Path getPathFromSourceExecPath(Path execRoot, PathFragment execPath);

  /**
   * Determines if an artifact is derived, that is, its root is a derived root or its exec path
   * starts with the bazel-out prefix.
   *
   * @param execPath The artifact's exec path.
   */
  boolean isDerivedArtifact(PathFragment execPath);
}
