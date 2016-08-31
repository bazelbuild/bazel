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
import javax.annotation.Nullable;

/**
 * A target that can provide the aar artifact of Android libraries and all the manifests that are
 * merged into the main aar manifest.
 */
@AutoValue
@Immutable
public abstract class AndroidLibraryAarProvider implements TransitiveInfoProvider {

  public static AndroidLibraryAarProvider create(@Nullable Aar aar, NestedSet<Aar> transitiveAars) {
    return new AutoValue_AndroidLibraryAarProvider(aar, transitiveAars);
  }

  @Nullable public abstract Aar getAar();

  public abstract NestedSet<Aar> getTransitiveAars();

  /** The .aar file and associated AndroidManifest.xml contributed by a single target. */
  @AutoValue
  @Immutable
  public abstract static class Aar {
    public static Aar create(Artifact aar, Artifact manifest) {
      return new AutoValue_AndroidLibraryAarProvider_Aar(aar, manifest);
    }

    public abstract Artifact getAar();

    public abstract Artifact getManifest();

    Aar() {}
  }

  AndroidLibraryAarProvider() {}
}
