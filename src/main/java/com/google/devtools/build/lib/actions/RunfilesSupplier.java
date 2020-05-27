// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Map;

/** Convenience wrapper around runfiles allowing lazy expansion. */
// TODO(bazel-team): Ideally we could refer to Runfiles objects directly here, but current package
// structure makes this difficult. Consider moving things around to make this possible.
//
// RunfilesSuppliers appear to be Starlark values;
// they are exposed through ctx.resolve_tools[2], for example.
public interface RunfilesSupplier extends StarlarkValue {

  /** @return the contained artifacts */
  NestedSet<Artifact> getArtifacts();

  /** @return the runfiles' root directories. */
  ImmutableSet<PathFragment> getRunfilesDirs();

  /**
   * Returns mappings from runfiles directories to artifact mappings in that directory.
   *
   * @return runfiles' mappings
   * @throws IOException
   */
  ImmutableMap<PathFragment, Map<PathFragment, Artifact>> getMappings();

  /** @return the runfiles manifest artifacts, if any. */
  ImmutableList<Artifact> getManifests();

  /**
   * Returns whether for a given {@code runfilesDir} the runfile symlinks are materialized during
   * build. Also returns {@code false} if the runfiles supplier doesn't know about the directory.
   *
   * @param runfilesDir runfiles directory relative to the exec root
   */
  boolean isBuildRunfileLinks(PathFragment runfilesDir);

  /**
   * Returns whether it's allowed to create runfile symlinks in the {@code runfilesDir}. Also
   * returns {@code false} if the runfiles supplier doesn't know about the directory.
   *
   * @param runfilesDir runfiles directory relative to the exec root
   */
  boolean isRunfileLinksEnabled(PathFragment runfilesDir);
}
