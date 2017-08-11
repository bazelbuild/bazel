// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.aapt2;

import java.nio.file.Path;

/**
 * Contains reference to the aapt2 generated .flat file archive and a manifest.
 *
 * <p>This represents the state between the aapt2 compile and link actions.
 */
public class CompiledResources {

  private final Path resources;
  private final Path manifest;

  private CompiledResources(Path resources, Path manifest) {
    this.resources = resources;
    this.manifest = manifest;
  }

  public static CompiledResources from(Path resources, Path manifest) {
    return new CompiledResources(resources, manifest);
  }

  public Path asZip() {
    return resources;
  }

  public Path asManifest() {
    return manifest;
  }
}
