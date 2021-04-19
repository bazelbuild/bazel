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
package com.google.devtools.build.lib.skyframe.serialization.testutils;

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.ArtifactResolver;
import com.google.devtools.build.lib.actions.ArtifactResolver.ArtifactResolverSupplier;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.PackageRootResolver;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.util.Map;
import javax.annotation.Nullable;

/** Utilities for testing with serialization dependencies. */
public class SerializationDepsUtils {

  /** Default serialization dependencies for testing. */
  public static final ImmutableClassToInstanceMap<?> SERIALIZATION_DEPS_FOR_TEST =
      ImmutableClassToInstanceMap.of(
          ArtifactResolverSupplier.class, new ArtifactResolverSupplierForTest());

  /**
   * An {@link ArtifactResolverSupplier} that calls directly into the {@link SourceArtifact}
   * constructor.
   */
  public static class ArtifactResolverSupplierForTest implements ArtifactResolverSupplier {

    @Override
    public ArtifactResolver get() {
      return new ArtifactResolver() {
        @Override
        public Artifact getSourceArtifact(PathFragment execPath, Root root, ArtifactOwner owner) {
          return new SourceArtifact(ArtifactRoot.asSourceRoot(root), execPath, owner);
        }

        @Override
        public Artifact getSourceArtifact(PathFragment execPath, Root root) {
          throw new UnsupportedOperationException();
        }

        @Override
        public Artifact resolveSourceArtifact(
            PathFragment execPath, RepositoryName repositoryName) {
          throw new UnsupportedOperationException();
        }

        @Nullable
        @Override
        public Map<PathFragment, Artifact> resolveSourceArtifacts(
            Iterable<PathFragment> execPaths, PackageRootResolver resolver) {
          throw new UnsupportedOperationException();
        }

        @Override
        public Path getPathFromSourceExecPath(Path execRoot, PathFragment execPath) {
          throw new UnsupportedOperationException();
        }
      };
    }

    @Override
    public Artifact.DerivedArtifact intern(Artifact.DerivedArtifact original) {
      return original;
    }
  }
}
