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
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcCompilationInfoApi;

/** Wrapper for every C++ compilation provider. */
@Immutable
@AutoCodec
public final class CcCompilationInfo extends NativeInfo implements CcCompilationInfoApi {
  public static final NativeProvider<CcCompilationInfo> PROVIDER =
      new NativeProvider<CcCompilationInfo>(CcCompilationInfo.class, "CcCompilationInfo") {};

  private final CcCompilationContext ccCompilationContext;

  @AutoCodec.Instantiator
  @VisibleForSerialization
  CcCompilationInfo(CcCompilationContext ccCompilationContext) {
    super(PROVIDER);
    this.ccCompilationContext = ccCompilationContext;
  }

  @Override
  public CcCompilationContext getCcCompilationContext() {
    return ccCompilationContext;
  }

  /** A Builder for {@link CcCompilationInfo}. */
  public static class Builder {
    CcCompilationContext ccCompilationContext;

    public static CcCompilationInfo.Builder create() {
      return new CcCompilationInfo.Builder();
    }

    public <P extends TransitiveInfoProvider> Builder setCcCompilationContext(
        CcCompilationContext ccCompilationContext) {
      Preconditions.checkState(this.ccCompilationContext == null);
      this.ccCompilationContext = ccCompilationContext;
      return this;
    }

    public CcCompilationInfo build() {
      return new CcCompilationInfo(ccCompilationContext);
    }
  }

  public static ImmutableList<CcCompilationContext> getCcCompilationContexts(
      Iterable<? extends TransitiveInfoCollection> deps) {
    ImmutableList.Builder<CcCompilationContext> ccCompilationContextsBuilder =
        ImmutableList.builder();
    for (CcCompilationInfo ccCompilationInfo :
        AnalysisUtils.getProviders(deps, CcCompilationInfo.PROVIDER)) {
      CcCompilationContext ccCompilationContext = ccCompilationInfo.getCcCompilationContext();
      if (ccCompilationContext != null) {
        ccCompilationContextsBuilder.add(ccCompilationContext);
      }
    }
    return ccCompilationContextsBuilder.build();
  }
}
