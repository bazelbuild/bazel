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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/** Wrapper for every C++ compilation provider. */
@Immutable
@AutoCodec
@SkylarkModule(
  name = "cc_compilation_info",
  documented = false,
  category = SkylarkModuleCategory.PROVIDER,
  doc = "Wrapper for every C++ compilation provider"
)
public final class CcCompilationInfo extends NativeInfo {
  public static final NativeProvider<CcCompilationInfo> PROVIDER =
      new NativeProvider<CcCompilationInfo>(CcCompilationInfo.class, "CcCompilationInfo") {};

  private final CcCompilationContextInfo ccCompilationContextInfo;

  @AutoCodec.Instantiator
  @VisibleForSerialization
  CcCompilationInfo(CcCompilationContextInfo ccCompilationContextInfo) {
    super(PROVIDER);
    this.ccCompilationContextInfo = ccCompilationContextInfo;
  }

  @SkylarkCallable(
    name = "cc_compilation_context_info",
    structField = true,
    allowReturnNones = true,
    doc = "Returns compilation information for this C++ target."
  )
  public CcCompilationContextInfo getCcCompilationContextInfo() {
    return ccCompilationContextInfo;
  }

  /** A Builder for {@link CcCompilationInfo}. */
  public static class Builder {
    CcCompilationContextInfo ccCompilationContextInfo;

    public static CcCompilationInfo.Builder create() {
      return new CcCompilationInfo.Builder();
    }

    public <P extends TransitiveInfoProvider> Builder setCcCompilationContextInfo(
        CcCompilationContextInfo ccCompilationContextInfo) {
      Preconditions.checkState(this.ccCompilationContextInfo == null);
      this.ccCompilationContextInfo = ccCompilationContextInfo;
      return this;
    }

    public CcCompilationInfo build() {
      return new CcCompilationInfo(ccCompilationContextInfo);
    }
  }

  public static ImmutableList<CcCompilationContextInfo> getCcCompilationContextInfos(
      Iterable<? extends TransitiveInfoCollection> deps) {
    ImmutableList.Builder<CcCompilationContextInfo> ccCompilationContextInfosBuilder =
        ImmutableList.builder();
    for (CcCompilationInfo ccCompilationInfo :
        AnalysisUtils.getProviders(deps, CcCompilationInfo.PROVIDER)) {
      CcCompilationContextInfo ccCompilationContextInfo =
          ccCompilationInfo.getCcCompilationContextInfo();
      if (ccCompilationContextInfo != null) {
        ccCompilationContextInfosBuilder.add(ccCompilationContextInfo);
      }
    }
    return ccCompilationContextInfosBuilder.build();
  }
}
