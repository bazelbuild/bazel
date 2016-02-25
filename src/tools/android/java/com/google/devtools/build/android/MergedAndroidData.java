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
package com.google.devtools.build.android;

import java.nio.file.Path;

/**
 * Represents the AndroidData before processing, after merging.
 * 
 * <p>
 * The life cycle of AndroidData goes:
 * <pre>
 * UnvalidatedAndroidData -> MergedAndroidData -> DensityFilteredAndroidData
 *      -> DependencyAndroidData
 * </pre>
 */
class MergedAndroidData {

  private Path resourceDir;
  private Path assetDir;
  private Path manifest;

  public MergedAndroidData(Path resources, Path assets, Path manifest) {
    this.resourceDir = resources;
    this.assetDir = assets;
    this.manifest = manifest;
  }

  public Path getResourceDir() {
    return resourceDir;
  }

  public Path getAssetDir() {
    return assetDir;
  }

  public Path getManifest() {
    return manifest;
  }

  public DensityFilteredAndroidData filter(
      DensitySpecificResourceFilter resourceFilter,
      DensitySpecificManifestProcessor manifestProcessor)
          throws ManifestProcessingException {
    return new DensityFilteredAndroidData(resourceFilter.filter(resourceDir),
        assetDir, manifestProcessor.process(manifest));
  }
}
