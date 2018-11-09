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
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/**
 * A {@link TransitiveInfoProvider} that describes how C++ FDO compilation should be done.
 *
 * <p><b>The {@code fdoProfilePath} member was a mistake. DO NOT USE IT FOR ANYTHING!</b>
 */
@Immutable
@AutoCodec
public class FdoProvider implements TransitiveInfoProvider {
  /**
   * The FDO mode we are operating in.
   */
  public enum FdoMode {
    /** FDO is turned off. */
    OFF,

    /** FDO based on automatically collected data. */
    AUTO_FDO,

    /** FDO based on cross binary collected data. */
    XBINARY_FDO,

    /** Instrumentation-based FDO implemented on LLVM. */
    LLVM_FDO,
  }

  private final FdoMode fdoMode;
  private final String fdoInstrument;
  private final Artifact profileArtifact;
  private final Artifact prefetchHintsArtifact;
  private final Artifact protoProfileArtifact;

  @AutoCodec.Instantiator
  public FdoProvider(
      FdoMode fdoMode,
      String fdoInstrument,
      Artifact profileArtifact,
      Artifact prefetchHintsArtifact,
      Artifact protoProfileArtifact) {
    this.fdoMode = fdoMode;
    this.fdoInstrument = fdoInstrument;
    this.profileArtifact = profileArtifact;
    this.prefetchHintsArtifact = prefetchHintsArtifact;
    this.protoProfileArtifact = protoProfileArtifact;
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

  public Artifact getProtoProfileArtifact() {
    return protoProfileArtifact;
  }
}
