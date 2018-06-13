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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcLinkingInfoApi;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.SkylarkType;

/** Wrapper for every C++ linking provider. */
@Immutable
@AutoCodec
public final class CcLinkingInfo extends NativeInfo implements CcLinkingInfoApi {

  private static final FunctionSignature.WithValues<Object, SkylarkType> SIGNATURE =
      FunctionSignature.WithValues.create(
          FunctionSignature.of(
              /* numMandatoryPositionals= */ 0,
              /* numOptionalPositionals= */ 0,
              /* numMandatoryNamedOnly= */ 1,
              /* starArg= */ false,
              /* kwArg= */ false,
              "cc_runfiles"),
          /* defaultValues= */ ImmutableList.of(),
          /* types= */ ImmutableList.of(SkylarkType.of(CcRunfiles.class)));

  public static final NativeProvider<CcLinkingInfo> PROVIDER =
      new NativeProvider<CcLinkingInfo>(CcLinkingInfo.class, "CcLinkingInfo", SIGNATURE) {
        @Override
        @SuppressWarnings("unchecked")
        protected CcLinkingInfo createInstanceFromSkylark(
            Object[] args, Environment env, Location loc) throws EvalException {
          CcCommon.checkLocationWhitelisted(loc);
          CcLinkingInfo.Builder ccLinkingInfoBuilder = CcLinkingInfo.Builder.create();
          ccLinkingInfoBuilder.setCcRunfiles((CcRunfiles) args[0]);
          return ccLinkingInfoBuilder.build();
        }
      };

  private final CcLinkParamsStore ccLinkParamsStore;
  private final CcRunfiles ccRunfiles;
  private final CcDynamicLibrariesForRuntime ccDynamicLibrariesForRuntime;

  @AutoCodec.Instantiator
  @VisibleForSerialization
  CcLinkingInfo(
      CcLinkParamsStore ccLinkParamsStore,
      CcRunfiles ccRunfiles,
      CcDynamicLibrariesForRuntime ccDynamicLibrariesForRuntime) {
    super(PROVIDER);
    this.ccLinkParamsStore = ccLinkParamsStore;
    this.ccRunfiles = ccRunfiles;
    this.ccDynamicLibrariesForRuntime = ccDynamicLibrariesForRuntime;
  }

  public CcLinkParamsStore getCcLinkParamsStore() {
    return ccLinkParamsStore;
  }

  @Override
  public CcRunfiles getCcRunfiles() {
    return ccRunfiles;
  }

  public CcDynamicLibrariesForRuntime getCcDynamicLibrariesForRuntime() {
    return ccDynamicLibrariesForRuntime;
  }

  /** A Builder for {@link CcLinkingInfo}. */
  public static class Builder {
    CcLinkParamsStore ccLinkParamsStore;
    CcRunfiles ccRunfiles;
    CcDynamicLibrariesForRuntime ccDynamicLibrariesForRuntime;

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

    public Builder setCcDynamicLibrariesForRuntime(
        CcDynamicLibrariesForRuntime ccDynamicLibrariesForRuntime) {
      Preconditions.checkState(this.ccDynamicLibrariesForRuntime == null);
      this.ccDynamicLibrariesForRuntime = ccDynamicLibrariesForRuntime;
      return this;
    }

    public CcLinkingInfo build() {
      return new CcLinkingInfo(ccLinkParamsStore, ccRunfiles, ccDynamicLibrariesForRuntime);
    }
  }
}
