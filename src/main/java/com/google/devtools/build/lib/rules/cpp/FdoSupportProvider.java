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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.cpp.FdoSupport.FdoMode;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/**
 * A {@link TransitiveInfoProvider} so that {@code cc_toolchain} can pass {@link FdoSupport} to the
 * C++ rules.
 */
@Immutable
@AutoCodec
public class FdoSupportProvider implements TransitiveInfoProvider {
  private final FdoSupport fdoSupport;
  private final FdoMode fdoMode;
  private final String fdoInstrument;
  private final Artifact profileArtifact;
  private final Artifact prefetchHintsArtifact;

  @AutoCodec.Instantiator
  public FdoSupportProvider(FdoSupport fdoSupport, FdoMode fdoMode, String fdoInstrument,
      Artifact profileArtifact, Artifact prefetchHintsArtifact) {
    this.fdoSupport = fdoSupport;
    this.fdoMode = fdoMode;
    this.fdoInstrument = fdoInstrument;
    this.profileArtifact = profileArtifact;
    this.prefetchHintsArtifact = prefetchHintsArtifact;
  }

  public FdoSupport getFdoSupport() {
    return fdoSupport;
  }

  public String getFdoInstrument() {
    return fdoInstrument;
  }

  public FdoMode getFdoMode() {
    return fdoMode;
  }

  public Artifact getProfileArtifact() {
    return profileArtifact;
  }

  public Artifact getPrefetchHintsArtifact() {
    return prefetchHintsArtifact;
  }
}
