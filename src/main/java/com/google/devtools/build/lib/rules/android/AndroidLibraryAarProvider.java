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
package com.google.devtools.build.lib.rules.android;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * A target that can provide the aar artifact of Android libraries and all the manifests that are
 * merged into the main aar manifest.
 */
@Immutable
public final class AndroidLibraryAarProvider implements TransitiveInfoProvider {

  private final Artifact aar;
  private final Artifact manifest;

  public AndroidLibraryAarProvider(Artifact aar, Artifact manifest) {
    this.aar = Preconditions.checkNotNull(aar);
    this.manifest = Preconditions.checkNotNull(manifest);
  }

  public Artifact getAar() {
    return aar;
  }

  public Artifact getManifest() {
    return manifest;
  }
}
