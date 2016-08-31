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
package com.google.devtools.build.lib.rules.android;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/** A provider for targets that can build .apk files. Currently used for coverage collection. */
@AutoValue
@Immutable
public abstract class ApkProvider implements TransitiveInfoProvider {

  public static ApkProvider create(
      NestedSet<Artifact> transitiveApks,
      NestedSet<Artifact> coverageMetdata,
      NestedSet<Artifact> mergedManifests) {
    return new AutoValue_ApkProvider(transitiveApks, coverageMetdata, mergedManifests);
  }

  /** Returns the APK files generated in the transitive closure. */
  public abstract NestedSet<Artifact> getTransitiveApks();

  /** Returns the coverage metadata artifacts generated in the transitive closure. */
  public abstract NestedSet<Artifact> getCoverageMetadata();

  /** Returns the merged manifests */
  public abstract NestedSet<Artifact> getMergedManifests();

  ApkProvider() {}
}
