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
public class FdoContext implements TransitiveInfoProvider {
  public static FdoContext getDisabledContext() {
    return new FdoContext(/* branchFdoProfile= */ null, /* prefetchHintsArtifact= */ null);
  }

  /** The Branch FDO mode we are operating in. */
  public enum BranchFdoMode {
    /** FDO based on automatically collected data. */
    AUTO_FDO,

    /** FDO based on cross binary collected data. */
    XBINARY_FDO,

    /** Instrumentation-based FDO implemented on LLVM. */
    LLVM_FDO,

    /** Instrumentation-based Context Sensitive FDO implemented on LLVM. */
    LLVM_CS_FDO,
  }

  /** A POJO encapsulating the branch profiling configuration. */
  @Immutable
  public static class BranchFdoProfile {
    private final BranchFdoMode branchFdoMode;
    private final Artifact profileArtifact;
    private final Artifact protoProfileArtifact;

    public BranchFdoProfile(
        BranchFdoMode branchFdoMode, Artifact profileArtifact, Artifact protoProfileArtifact) {
      this.branchFdoMode = branchFdoMode;
      this.profileArtifact = profileArtifact;
      this.protoProfileArtifact = protoProfileArtifact;
    }

    public boolean isAutoFdo() {
      return branchFdoMode == BranchFdoMode.AUTO_FDO;
    }

    public boolean isAutoXBinaryFdo() {
      return branchFdoMode == BranchFdoMode.XBINARY_FDO;
    }

    public boolean isLlvmFdo() {
      return branchFdoMode == BranchFdoMode.LLVM_FDO;
    }

    public boolean isLlvmCSFdo() {
      return branchFdoMode == BranchFdoMode.LLVM_CS_FDO;
    }

    public Artifact getProfileArtifact() {
      return profileArtifact;
    }

    public Artifact getProtoProfileArtifact() {
      return protoProfileArtifact;
    }
  }

  private final BranchFdoProfile branchFdoProfile;
  private final Artifact prefetchHintsArtifact;

  @AutoCodec.Instantiator
  public FdoContext(BranchFdoProfile branchFdoProfile, Artifact prefetchHintsArtifact) {
    this.branchFdoProfile = branchFdoProfile;
    this.prefetchHintsArtifact = prefetchHintsArtifact;
  }

  public BranchFdoProfile getBranchFdoProfile() {
    return branchFdoProfile;
  }

  public Artifact getPrefetchHintsArtifact() {
    return prefetchHintsArtifact;
  }

  boolean hasArtifacts(CppConfiguration cppConfiguration) {
    if (cppConfiguration.isToolConfigurationDoNotUseWillBeRemovedFor129045294()) {
      // We don't want FDO for host configuration
      return false;
    }
    return getBranchFdoProfile() != null || getPrefetchHintsArtifact() != null;
  }
}
