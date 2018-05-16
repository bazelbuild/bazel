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

  private final CcLinkParamsStore ccLinkParamsStore;
  private final CcRunfiles ccRunfiles;
  private final CcExecutionDynamicLibraries ccExecutionDynamicLibraries;

  @AutoCodec.Instantiator
  @VisibleForSerialization
  CcLinkingInfo(
      CcLinkParamsStore ccLinkParamsStore,
      CcRunfiles ccRunfiles,
      CcExecutionDynamicLibraries ccExecutionDynamicLibraries) {
    super(PROVIDER);
    this.ccLinkParamsStore = ccLinkParamsStore;
    this.ccRunfiles = ccRunfiles;
    this.ccExecutionDynamicLibraries = ccExecutionDynamicLibraries;
  }

  public CcLinkParamsStore getCcLinkParamsStore() {
    return ccLinkParamsStore;
  }

  public CcRunfiles getCcRunfiles() {
    return ccRunfiles;
  }

  public CcExecutionDynamicLibraries getCcExecutionDynamicLibraries() {
    return ccExecutionDynamicLibraries;
  }

  /** A Builder for {@link CcLinkingInfo}. */
  public static class Builder {
    CcLinkParamsStore ccLinkParamsStore;
    CcRunfiles ccRunfiles;
    CcExecutionDynamicLibraries ccExecutionDynamicLibraries;

    public static CcLinkingInfo.Builder create() {
      return new CcLinkingInfo.Builder();
    }

    public Builder setCcLinkParamsStore(CcLinkParamsStore ccLinkParamsStore) {
      Preconditions.checkState(this.ccLinkParamsStore == null);
      this.ccLinkParamsStore = ccLinkParamsStore;
      return this;
    }

    public Builder setCcRunfiles(CcRunfiles ccRunfiles) {
      Preconditions.checkState(this.ccRunfiles == null);
      this.ccRunfiles = ccRunfiles;
      return this;
    }

    public Builder setCcExecutionDynamicLibraries(
        CcExecutionDynamicLibraries ccExecutionDynamicLibraries) {
      Preconditions.checkState(this.ccExecutionDynamicLibraries == null);
      this.ccExecutionDynamicLibraries = ccExecutionDynamicLibraries;
      return this;
    }

    public CcLinkingInfo build() {
      return new CcLinkingInfo(ccLinkParamsStore, ccRunfiles, ccExecutionDynamicLibraries);
    }
  }
}
