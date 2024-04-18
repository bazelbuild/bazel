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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.cpp.FdoContext.BranchFdoProfile;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.BranchFdoProfileApi;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.FdoContextApi;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkThread;

/**
 * Describes how C++ FDO compilation should be done.
 *
 * <p><b>The {@code fdoProfilePath} member was a mistake. DO NOT USE IT FOR ANYTHING!</b>
 */
@Immutable
public final class FdoContext implements FdoContextApi<BranchFdoProfile> {
  public static FdoContext getDisabledContext() {
    return new FdoContext(
        /* branchFdoProfile= */ null,
        /* prefetchHintsArtifact= */ null,
        /* propellerOptimizeInputFile= */ null,
        /* memprofProfileArtifact= */ null);
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
  public static class BranchFdoProfile implements BranchFdoProfileApi {
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

    @Override
    public boolean isAutoFdoForStarlark(StarlarkThread thread) throws EvalException {
      CcModule.checkPrivateStarlarkificationAllowlist(thread);
      return isAutoFdo();
    }

    public boolean isAutoXBinaryFdo() {
      return branchFdoMode == BranchFdoMode.XBINARY_FDO;
    }

    @Override
    public boolean isAutoXBinaryFdoForStarlark(StarlarkThread thread) throws EvalException {
      CcModule.checkPrivateStarlarkificationAllowlist(thread);
      return isAutoXBinaryFdo();
    }

    public boolean isLlvmFdo() {
      return branchFdoMode == BranchFdoMode.LLVM_FDO;
    }

    @Override
    public boolean isLlvmFdoForStarlark(StarlarkThread thread) throws EvalException {
      CcModule.checkPrivateStarlarkificationAllowlist(thread);
      return isLlvmFdo();
    }

    public boolean isLlvmCSFdo() {
      return branchFdoMode == BranchFdoMode.LLVM_CS_FDO;
    }

    @Override
    public boolean isLlvmCSFdoForStarlark(StarlarkThread thread) throws EvalException {
      CcModule.checkPrivateStarlarkificationAllowlist(thread);
      return isLlvmCSFdo();
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
  private final PropellerOptimizeInputFile propellerOptimizeInputFile;
  private final Artifact memprofProfileArtifact;

  public FdoContext(
      BranchFdoProfile branchFdoProfile,
      Artifact prefetchHintsArtifact,
      PropellerOptimizeInputFile propellerOptimizeInputFile,
      Artifact memprofProfileArtifact) {
    this.branchFdoProfile = branchFdoProfile;
    this.prefetchHintsArtifact = prefetchHintsArtifact;
    this.propellerOptimizeInputFile = propellerOptimizeInputFile;
    this.memprofProfileArtifact = memprofProfileArtifact;
  }

  public BranchFdoProfile getBranchFdoProfile() {
    return branchFdoProfile;
  }

  @Override
  @Nullable
  public BranchFdoProfile getBranchFdoProfileForStarlark(StarlarkThread thread)
      throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return branchFdoProfile;
  }

  public Artifact getPrefetchHintsArtifact() {
    return prefetchHintsArtifact;
  }

  public PropellerOptimizeInputFile getPropellerOptimizeInputFile() {
    return propellerOptimizeInputFile;
  }

  public Artifact getMemProfProfileArtifact() {
    return memprofProfileArtifact;
  }

  boolean hasArtifacts(CppConfiguration cppConfiguration) {
    return getBranchFdoProfile() != null
        || getPrefetchHintsArtifact() != null
        || getPropellerOptimizeInputFile() != null
        || getMemProfProfileArtifact() != null;
  }
}
