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
import com.google.devtools.build.lib.packages.StructImpl;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/**
 * Describes how C++ FDO compilation should be done.
 *
 * <p>A POJO encapsulating the branch profiling configuration. For implementation see
 * fdo_context.bzl.
 *
 * <p><b>The {@code fdoProfilePath} member was a mistake. DO NOT USE IT FOR ANYTHING!</b>
 */
@Immutable
public final class FdoContext {
  private final StructImpl fdoContextStruct;

  /** A POJO encapsulating the branch profiling configuration. */
  @Immutable
  public static class BranchFdoProfile {
    private final StructImpl branchFdoProfile;

    public BranchFdoProfile(StructImpl branchFdoProfile) {
      this.branchFdoProfile = branchFdoProfile;
    }

    public boolean isAutoFdo() throws EvalException {
      return getBranchFdoMode().equals("auto_fdo");
    }

    public boolean isAutoXBinaryFdo() throws EvalException {
      return getBranchFdoMode().equals("xbinary_fdo");
    }

    public boolean isLlvmFdo() throws EvalException {
      return getBranchFdoMode().equals("llvm_fdo");
    }

    public boolean isLlvmCSFdo() throws EvalException {
      return getBranchFdoMode().equals("llvm_cs_fdo");
    }

    @Nullable
    public Artifact getProfileArtifact() throws EvalException {
      return branchFdoProfile.getNoneableValue("profile_artifact", Artifact.class);
    }

    @Nullable
    public Artifact getProtoProfileArtifact() throws EvalException {
      return branchFdoProfile.getNoneableValue("proto_profile_artifact", Artifact.class);
    }

    private String getBranchFdoMode() throws EvalException {
      return branchFdoProfile.getValue("branch_fdo_mode", String.class);
    }
  }

  public FdoContext(StructImpl fdoContextStruct) {
    this.fdoContextStruct = fdoContextStruct;
  }

  @Nullable
  public BranchFdoProfile getBranchFdoProfile() throws EvalException {
    StructImpl branchFdoProfile =
        fdoContextStruct.getNoneableValue("branch_fdo_profile", StructImpl.class);
    if (branchFdoProfile == null) {
      return null;
    }
    return new BranchFdoProfile(branchFdoProfile);
  }

  public Artifact getPrefetchHintsArtifact() throws EvalException {
    return fdoContextStruct.getNoneableValue("prefetch_hints_artifact", Artifact.class);
  }

  @Nullable
  public PropellerOptimizeInputFile getPropellerOptimizeInputFile() throws EvalException {
    StructImpl inputFile =
        fdoContextStruct.getNoneableValue("propeller_optimize_info", StructImpl.class);
    if (inputFile == null) {
      return null;
    }
    return new PropellerOptimizeInputFile(inputFile);
  }

  public Artifact getMemProfProfileArtifact() throws EvalException {
    return fdoContextStruct.getNoneableValue("memprof_profile_artifact", Artifact.class);
  }

  boolean hasArtifacts(CppConfiguration cppConfiguration) throws EvalException {
    return getBranchFdoProfile() != null
        || getPrefetchHintsArtifact() != null
        || getPropellerOptimizeInputFile() != null
        || getMemProfProfileArtifact() != null;
  }
}
