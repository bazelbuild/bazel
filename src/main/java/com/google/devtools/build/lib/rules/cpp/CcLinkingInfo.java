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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/** Wrapper for every C++ linking provider. */
@Immutable
@AutoCodec
@SkylarkModule(
  name = "cc_linking_info",
  documented = false,
  category = SkylarkModuleCategory.PROVIDER,
  doc = "Wrapper for every C++ linking provider"
)
public final class CcLinkingInfo extends NativeInfo {
  public static final NativeProvider<CcLinkingInfo> PROVIDER =
      new NativeProvider<CcLinkingInfo>(CcLinkingInfo.class, "CcLinkingInfo") {};

  private final CcLinkParamsInfo ccLinkParamsInfo;

  @AutoCodec.Instantiator
  @VisibleForSerialization
  CcLinkingInfo(CcLinkParamsInfo ccLinkParamsInfo) {
    super(PROVIDER);
    this.ccLinkParamsInfo = ccLinkParamsInfo;
  }

  public CcLinkParamsInfo getCcLinkParamsInfo() {
    return ccLinkParamsInfo;
  }

  /** A Builder for {@link CcLinkingInfo}. */
  public static class Builder {
    CcLinkParamsInfo ccLinkParamsInfo;

    public static CcLinkingInfo.Builder create() {
      return new CcLinkingInfo.Builder();
    }

    public Builder setCcLinkParamsInfo(CcLinkParamsInfo ccLinkParamsInfo) {
      Preconditions.checkState(this.ccLinkParamsInfo == null);
      this.ccLinkParamsInfo = ccLinkParamsInfo;
      return this;
    }

    public CcLinkingInfo build() {
      return new CcLinkingInfo(ccLinkParamsInfo);
    }
  }

  public static ImmutableList<CcLinkParamsInfo> getCcLinkParamsInfos(
      Iterable<? extends TransitiveInfoCollection> deps) {
    ImmutableList.Builder<CcLinkParamsInfo> ccLinkParamsInfosBuilder = ImmutableList.builder();
    for (CcLinkingInfo ccLinkingInfo : AnalysisUtils.getProviders(deps, CcLinkingInfo.PROVIDER)) {
      CcLinkParamsInfo ccLinkParamsInfo = ccLinkingInfo.getCcLinkParamsInfo();
      if (ccLinkParamsInfo != null) {
        ccLinkParamsInfosBuilder.add(ccLinkParamsInfo);
      }
    }
    return ccLinkParamsInfosBuilder.build();
  }
}
