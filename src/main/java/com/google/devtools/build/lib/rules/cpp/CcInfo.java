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
import com.google.devtools.build.lib.rules.cpp.LibraryToLink.CcLinkingContext;
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
  private final CcLinkingContext ccLinkingContext;

  public CcInfo(CcCompilationContext ccCompilationContext, CcLinkingContext ccLinkingContext) {
    super(PROVIDER);
    this.ccCompilationContext = ccCompilationContext;
    this.ccLinkingContext = ccLinkingContext;
  }

  @Override
  public CcCompilationContext getCcCompilationContext() {
    return ccCompilationContext;
  }

  @Override
  public CcLinkingContext getCcLinkingContext() {
    return ccLinkingContext;
  }

  public static CcInfo merge(Collection<CcInfo> ccInfos) {
    ImmutableList.Builder<CcCompilationContext> ccCompilationContexts = ImmutableList.builder();
    ImmutableList.Builder<CcLinkingContext> ccLinkingContexts = ImmutableList.builder();
    for (CcInfo ccInfo : ccInfos) {
      ccCompilationContexts.add(ccInfo.getCcCompilationContext());
      ccLinkingContexts.add(ccInfo.getCcLinkingContext());
    }
    CcCompilationContext.Builder builder =
        new CcCompilationContext.Builder(
            /* actionConstructionContext= */ null, /* configuration= */ null, /* label= */ null);

    return new CcInfo(
        builder.mergeDependentCcCompilationContexts(ccCompilationContexts.build()).build(),
        CcLinkingContext.merge(ccLinkingContexts.build()));
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
        || !this.getCcLinkingContext().equals(other.getCcLinkingContext())) {
      return false;
    }
    return true;
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(ccCompilationContext, ccLinkingContext);
  }

  public static Builder builder() {
    return new Builder();
  }

  /** A Builder for {@link CcInfo}. */
  public static class Builder {
    private CcCompilationContext ccCompilationContext;
    private CcLinkingContext ccLinkingContext;

    public CcInfo.Builder setCcCompilationContext(CcCompilationContext ccCompilationContext) {
      Preconditions.checkState(this.ccCompilationContext == null);
      this.ccCompilationContext = ccCompilationContext;
      return this;
    }

    public CcInfo.Builder setCcLinkingContext(CcLinkingContext ccLinkingContext) {
      Preconditions.checkState(this.ccLinkingContext == null);
      this.ccLinkingContext = ccLinkingContext;
      return this;
    }

    public CcInfo build() {
      if (ccCompilationContext == null) {
        ccCompilationContext = CcCompilationContext.EMPTY;
      }
      if (ccLinkingContext == null) {
        ccLinkingContext = CcLinkingContext.EMPTY;
      }
      return new CcInfo(ccCompilationContext, ccLinkingContext);
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
      // TODO(b/118663806): Eventually only CcLinkingContext will be allowed, this is for
      // backwards compatibility.
      CcLinkingContext ccLinkingContext = nullIfNone(skylarkCcLinkingInfo, CcLinkingContext.class);
      CcInfo.Builder ccInfoBuilder = CcInfo.builder();
      if (ccCompilationContext != null) {
        ccInfoBuilder.setCcCompilationContext(ccCompilationContext);
      }
      if (ccLinkingContext != null) {
        ccInfoBuilder.setCcLinkingContext(ccLinkingContext);
      }
      return ccInfoBuilder.build();
    }

    @Nullable
    private static <T> T nullIfNone(Object object, Class<T> type) {
      return object != Runtime.NONE ? type.cast(object) : null;
    }
  }
}
