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

import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.rules.cpp.LibraryToLinkWrapper.CcLinkingContext;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcInfoApi;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Runtime;
import java.util.Collection;
import javax.annotation.Nullable;

/** Provider for C++ compilation and linking information. */
@Immutable
public final class CcInfo extends NativeInfo implements CcInfoApi {
  public static final Provider PROVIDER = new Provider();
  public static final CcInfo EMPTY = CcInfo.builder().build();

  private final CcCompilationContext ccCompilationContext;
  // TODO(b/117875295): Rename CcLinkingInfo to CcLinkingContext
  private final CcLinkingInfo ccLinkingInfo;

  public CcInfo(CcCompilationContext ccCompilationContext, CcLinkingInfo ccLinkingInfo) {
    super(PROVIDER);
    this.ccCompilationContext = ccCompilationContext;
    this.ccLinkingInfo = ccLinkingInfo;
  }

  @Override
  public CcCompilationContext getCcCompilationContext() {
    return ccCompilationContext;
  }

  @Override
  public CcLinkingInfo getCcLinkingInfo() {
    return ccLinkingInfo;
  }

  public static CcInfo merge(Collection<CcInfo> ccInfos) {
    ImmutableList.Builder<CcCompilationContext> ccCompilationContexts = ImmutableList.builder();
    ImmutableList.Builder<CcLinkingInfo> ccLinkingInfos = ImmutableList.builder();
    for (CcInfo ccInfo : ccInfos) {
      ccCompilationContexts.add(ccInfo.getCcCompilationContext());
      ccLinkingInfos.add(ccInfo.getCcLinkingInfo());
    }
    CcCompilationContext.Builder builder =
        new CcCompilationContext.Builder(/* ruleContext= */ null);

    return new CcInfo(
        builder.mergeDependentCcCompilationContexts(ccCompilationContexts.build()).build(),
        CcLinkingInfo.merge(ccLinkingInfos.build()));
  }

  @Override
  public boolean equals(Object otherObject) {
    if (!(otherObject instanceof CcInfo)) {
      return false;
    }
    CcInfo other = (CcInfo) otherObject;
    if (this == other) {
      return true;
    }
    if (!this.ccCompilationContext.equals(other.ccCompilationContext)
        || !this.ccLinkingInfo.equals(other.ccLinkingInfo)) {
      return false;
    }
    return true;
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(ccCompilationContext, ccLinkingInfo);
  }

  public static Builder builder() {
    return new Builder();
  }

  /** A Builder for {@link CcInfo}. */
  public static class Builder {
    private CcCompilationContext ccCompilationContext;
    private CcLinkingInfo ccLinkingInfo;

    public CcInfo.Builder setCcCompilationContext(CcCompilationContext ccCompilationContext) {
      Preconditions.checkState(this.ccCompilationContext == null);
      this.ccCompilationContext = ccCompilationContext;
      return this;
    }

    public CcInfo.Builder setCcLinkingInfo(CcLinkingInfo ccLinkingInfo) {
      Preconditions.checkState(this.ccLinkingInfo == null);
      this.ccLinkingInfo = ccLinkingInfo;
      return this;
    }

    public CcInfo build() {
      if (ccCompilationContext == null) {
        ccCompilationContext = CcCompilationContext.EMPTY;
      }
      if (ccLinkingInfo == null) {
        ccLinkingInfo = CcLinkingInfo.EMPTY;
      }
      return new CcInfo(ccCompilationContext, ccLinkingInfo);
    }
  }

  /** Provider class for {@link CcInfo} objects. */
  public static class Provider extends BuiltinProvider<CcInfo> implements CcInfoApi.Provider {
    private Provider() {
      super(CcInfoApi.NAME, CcInfo.class);
    }

    @Override
    public CcInfoApi createInfo(
        Object skylarkCcCompilationContext,
        Object skylarkCcLinkingInfo,
        Location location,
        Environment environment)
        throws EvalException {
      CcCompilationContext ccCompilationContext =
          nullIfNone(skylarkCcCompilationContext, CcCompilationContext.class);
      CcLinkingInfo ccLinkingInfo = null;
      try {
        ccLinkingInfo = nullIfNone(skylarkCcLinkingInfo, CcLinkingInfo.class);
      } catch (ClassCastException e) {
        // TODO(b/118663806): Eventually only CcLinkingContext will be allowed, this is for
        // backwards compatibility.
        CcLinkingContext ccLinkingContext =
            nullIfNone(skylarkCcLinkingInfo, CcLinkingContext.class);
        if (ccLinkingContext != null) {
          ccLinkingInfo = ccLinkingContext.toCcLinkingInfo();
        }
      }
      CcInfo.Builder ccInfoBuilder = CcInfo.builder();
      if (ccCompilationContext != null) {
        ccInfoBuilder.setCcCompilationContext(ccCompilationContext);
      }
      if (ccLinkingInfo != null) {
        ccInfoBuilder.setCcLinkingInfo(ccLinkingInfo);
      }
      return ccInfoBuilder.build();
    }

    @Nullable
    private static <T> T nullIfNone(Object object, Class<T> type) {
      return object != Runtime.NONE ? type.cast(object) : null;
    }
  }
}
