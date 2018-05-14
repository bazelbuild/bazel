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
package com.google.devtools.build.lib.rules.android;

import com.google.devtools.build.lib.actions.Artifact;
import javax.annotation.Nullable;

/**
 * A {@link MergableAndroidData} that may also contain a compiled symbols file.
 *
 * <p>TODO(b/76418178): Once resources and assets are completely decoupled and {@link
 * ResourceContainer} is removed, this interface can be replaced with {@link ParsedAndroidResources}
 */
public interface CompiledMergableAndroidData extends MergableAndroidData {
  @Nullable
  Artifact getCompiledSymbols();

  Iterable<Artifact> getArtifacts();

  Artifact getManifest();

  boolean isManifestExported();
}
