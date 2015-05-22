// Copyright 2015 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.util.Map;
import java.util.Map.Entry;

/**
 * {@link RunfilesSupplier} implementation wrapping directory to {@link Runfiles} objects mappings.
 */
public class RunfilesSupplierImpl implements RunfilesSupplier {

  private final ImmutableMap<PathFragment, Runfiles> inputRunfiles;

  /**
   * Create an instance when you have a a single mapping.
   *
   * @param runfilesDir the desired runfiles directory. Should be relative.
   * @param runfiles the runfiles for runfilesDir
   */
  public RunfilesSupplierImpl(PathFragment runfilesDir, Runfiles runfiles) {
    this.inputRunfiles = ImmutableMap.of(runfilesDir, runfiles);
  }

  @VisibleForTesting
  public RunfilesSupplierImpl(Map<PathFragment, Runfiles> inputRunfiles) {
    this.inputRunfiles = ImmutableMap.copyOf(inputRunfiles);
  }

  @Override
  public Iterable<Artifact> getArtifacts() {
    ImmutableSet.Builder<Artifact> builder = ImmutableSet.builder();
    for (Entry<PathFragment, Runfiles> entry : inputRunfiles.entrySet()) {
      // TODO(bazel-team): We can likely do without middlemen here, but we should filter that at
      // the Runfiles level.
      builder.addAll(entry.getValue().getAllArtifacts());
    }
    return builder.build();
  }

  @Override
  public ImmutableSet<PathFragment> getRunfilesDirs() {
    return inputRunfiles.keySet();
  }

  @Override
  public ImmutableMap<PathFragment, Map<PathFragment, Artifact>> getMappings() throws IOException {
    ImmutableMap.Builder<PathFragment, Map<PathFragment, Artifact>> result =
        ImmutableMap.builder();
    for (Entry<PathFragment, Runfiles> entry : inputRunfiles.entrySet()) {
      result.put(entry.getKey(), entry.getValue().getRunfilesInputs(null, null));
    }
    return result.build();
  }

}
