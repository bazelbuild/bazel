// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Merged Android Data that has yet to written into a {@link MergedAndroidData}.
 */
public class UnwrittenMergedAndroidData {

  private final Path manifest;
  private final AndroidDataSet resources;
  private final AndroidDataSet deps;

  public static UnwrittenMergedAndroidData of(
      Path manifest, AndroidDataSet resources, AndroidDataSet deps) {
    return new UnwrittenMergedAndroidData(manifest, resources, deps);
  }

  private UnwrittenMergedAndroidData(Path manifest, AndroidDataSet resources, AndroidDataSet deps) {
    this.manifest = manifest;
    this.resources = resources;
    this.deps = deps;
  }

  /**
   * Writes the android data to directories for consumption by aapt.
   * @param newResourceDirectory The new resource directory to write to.
   * @return A MergedAndroidData that is ready for further tool processing.
   * @throws IOException when something goes wrong while writing.
   */
  public MergedAndroidData write(Path newResourceDirectory) throws IOException {
    // TODO(corysmith): Implement write.
    throw new UnsupportedOperationException();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("manifest", manifest)
        .add("resources", resources)
        .add("deps", deps)
        .toString();
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof UnwrittenMergedAndroidData)) {
      return false;
    }
    UnwrittenMergedAndroidData that = (UnwrittenMergedAndroidData) other;
    return Objects.equals(manifest, that.manifest)
        && Objects.equals(resources, that.resources)
        && Objects.equals(deps, that.deps);
  }

  @Override
  public int hashCode() {
    return Objects.hash(manifest, resources, deps);
  }

  @VisibleForTesting
  Path getManifest() {
    return manifest;
  }

  @VisibleForTesting
  AndroidDataSet getResources() {
    return resources;
  }

  @VisibleForTesting
  AndroidDataSet getDeps() {
    return deps;
  }
}
