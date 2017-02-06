// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * A {@link TransitiveInfoProvider} so that {@code cc_toolchain} can pass {@link FdoSupport} to the
 * C++ rules.
 */
@Immutable
public class FdoSupportProvider implements TransitiveInfoProvider {
  private final FdoSupport fdoSupport;
  private final Artifact profileArtifact;
  // We *probably* don't need both an AutoFDO profile artifact and a profile artifact. However,
  // the decision whether to make the latter seems to depend on the feature configuration of the
  // eventual cc_binary rule, which we don't have in cc_toolchain, so we just create both because
  // one extra artifact doesn't harm anyone.
  private final Artifact autoProfileArtifact;
  private final ImmutableMap<PathFragment, Artifact> gcdaArtifacts;

  public FdoSupportProvider(FdoSupport fdoSupport, Artifact profileArtifact,
      Artifact autoProfileArtifact, ImmutableMap<PathFragment, Artifact> gcdaArtifacts) {
    this.fdoSupport = fdoSupport;
    this.profileArtifact = profileArtifact;
    this.autoProfileArtifact = autoProfileArtifact;
    this.gcdaArtifacts = gcdaArtifacts;
  }

  public FdoSupport getFdoSupport() {
    return fdoSupport;
  }

  public Artifact getProfileArtifact() {
    return profileArtifact;
  }

  public Artifact getAutoProfileArtifact() {
    return autoProfileArtifact;
  }

  public ImmutableMap<PathFragment, Artifact> getGcdaArtifacts() {
    return gcdaArtifacts;
  }
}
