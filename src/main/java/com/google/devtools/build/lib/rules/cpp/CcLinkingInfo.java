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
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.cpp.CcLinkParams.LinkOptions;
import com.google.devtools.build.lib.rules.cpp.LibraryToLinkWrapper.CcLinkingContext;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcLinkingInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.LibraryToLinkWrapperApi;
import com.google.devtools.build.lib.syntax.SkylarkList;
import java.util.Collection;

/** Wrapper for every C++ linking provider. */
@Immutable
@AutoCodec
public final class CcLinkingInfo implements CcLinkingInfoApi {

  public static final CcLinkingInfo EMPTY =
      CcLinkingInfo.Builder.create()
          .setStaticModeParamsForDynamicLibrary(CcLinkParams.EMPTY)
          .setStaticModeParamsForExecutable(CcLinkParams.EMPTY)
          .setDynamicModeParamsForDynamicLibrary(CcLinkParams.EMPTY)
          .setDynamicModeParamsForExecutable(CcLinkParams.EMPTY)
          .build();

  private final CcLinkParams staticModeParamsForExecutable;
  private final CcLinkParams staticModeParamsForDynamicLibrary;
  private final CcLinkParams dynamicModeParamsForExecutable;
  private final CcLinkParams dynamicModeParamsForDynamicLibrary;

  @AutoCodec.Instantiator
  @VisibleForSerialization
  CcLinkingInfo(
      CcLinkParams staticModeParamsForExecutable,
      CcLinkParams staticModeParamsForDynamicLibrary,
      CcLinkParams dynamicModeParamsForExecutable,
      CcLinkParams dynamicModeParamsForDynamicLibrary) {
    this.staticModeParamsForExecutable = staticModeParamsForExecutable;
    this.staticModeParamsForDynamicLibrary = staticModeParamsForDynamicLibrary;
    this.dynamicModeParamsForExecutable = dynamicModeParamsForExecutable;
    this.dynamicModeParamsForDynamicLibrary = dynamicModeParamsForDynamicLibrary;
  }

  @Override
  public CcLinkParams getStaticModeParamsForExecutable() {
    return staticModeParamsForExecutable;
  }

  @Override
  public CcLinkParams getStaticModeParamsForDynamicLibrary() {
    return staticModeParamsForDynamicLibrary;
  }

  @Override
  public CcLinkParams getDynamicModeParamsForExecutable() {
    return dynamicModeParamsForExecutable;
  }

  @Override
  public CcLinkParams getDynamicModeParamsForDynamicLibrary() {
    return dynamicModeParamsForDynamicLibrary;
  }

  @Override
  public SkylarkList<String> getSkylarkUserLinkFlags() {
    CcLinkingContext ccLinkingContext = LibraryToLinkWrapper.fromCcLinkingInfo(this);
    return SkylarkList.createImmutable(
        Streams.stream(ccLinkingContext.getUserLinkFlags())
            .map(LinkOptions::get)
            .flatMap(Collection::stream)
            .collect(ImmutableList.toImmutableList()));
  }

  @Override
  public SkylarkList<LibraryToLinkWrapperApi> getSkylarkLibrariesToLink() {
    CcLinkingContext ccLinkingContext = LibraryToLinkWrapper.fromCcLinkingInfo(this);
    return SkylarkList.createImmutable(ccLinkingContext.getLibraries().toList());
  }

  public static CcLinkingInfo merge(Collection<CcLinkingInfo> ccLinkingInfos) {
    CcLinkParams.Builder staticModeParamsForDynamicLibraryBuilder = CcLinkParams.builder();
    CcLinkParams.Builder staticModeParamsForExecutableBuilder = CcLinkParams.builder();
    CcLinkParams.Builder dynamicModeParamsForDynamicLibraryBuilder = CcLinkParams.builder();
    CcLinkParams.Builder dynamicModeParamsForExecutableBuilder = CcLinkParams.builder();
    for (CcLinkingInfo ccLinkingInfo : ccLinkingInfos) {
      staticModeParamsForDynamicLibraryBuilder.addTransitiveArgs(
          ccLinkingInfo.getStaticModeParamsForDynamicLibrary());
      staticModeParamsForExecutableBuilder.addTransitiveArgs(
          ccLinkingInfo.getStaticModeParamsForExecutable());
      dynamicModeParamsForDynamicLibraryBuilder.addTransitiveArgs(
          ccLinkingInfo.getDynamicModeParamsForDynamicLibrary());
      dynamicModeParamsForExecutableBuilder.addTransitiveArgs(
          ccLinkingInfo.getDynamicModeParamsForExecutable());
    }
    return new CcLinkingInfo.Builder()
        .setStaticModeParamsForDynamicLibrary(staticModeParamsForDynamicLibraryBuilder.build())
        .setStaticModeParamsForExecutable(staticModeParamsForExecutableBuilder.build())
        .setDynamicModeParamsForDynamicLibrary(dynamicModeParamsForDynamicLibraryBuilder.build())
        .setDynamicModeParamsForExecutable(dynamicModeParamsForExecutableBuilder.build())
        .build();
  }

  public CcLinkParams getCcLinkParams(boolean staticMode, boolean forDynamicLibrary) {
    if (staticMode) {
      if (forDynamicLibrary) {
        return getStaticModeParamsForDynamicLibrary();
      } else {
        return getStaticModeParamsForExecutable();
      }
    } else {
      if (forDynamicLibrary) {
        return getDynamicModeParamsForDynamicLibrary();
      } else {
        return getDynamicModeParamsForExecutable();
      }
    }
  }

  /** A Builder for {@link CcLinkingInfo}. */
  public static class Builder {
    CcLinkParams staticModeParamsForDynamicLibrary;
    CcLinkParams staticModeParamsForExecutable;
    CcLinkParams dynamicModeParamsForDynamicLibrary;
    CcLinkParams dynamicModeParamsForExecutable;

    public static CcLinkingInfo.Builder create() {
      return new CcLinkingInfo.Builder();
    }

    public Builder setStaticModeParamsForDynamicLibrary(CcLinkParams ccLinkParams) {
      Preconditions.checkState(this.staticModeParamsForDynamicLibrary == null);
      this.staticModeParamsForDynamicLibrary = ccLinkParams;
      return this;
    }

    public Builder setStaticModeParamsForExecutable(CcLinkParams ccLinkParams) {
      Preconditions.checkState(this.staticModeParamsForExecutable == null);
      this.staticModeParamsForExecutable = ccLinkParams;
      return this;
    }

    public Builder setDynamicModeParamsForDynamicLibrary(CcLinkParams ccLinkParams) {
      Preconditions.checkState(this.dynamicModeParamsForDynamicLibrary == null);
      this.dynamicModeParamsForDynamicLibrary = ccLinkParams;
      return this;
    }

    public Builder setDynamicModeParamsForExecutable(CcLinkParams ccLinkParams) {
      Preconditions.checkState(this.dynamicModeParamsForExecutable == null);
      this.dynamicModeParamsForExecutable = ccLinkParams;
      return this;
    }

    public CcLinkingInfo build() {
      Preconditions.checkNotNull(staticModeParamsForExecutable);
      Preconditions.checkNotNull(staticModeParamsForDynamicLibrary);
      Preconditions.checkNotNull(dynamicModeParamsForExecutable);
      Preconditions.checkNotNull(dynamicModeParamsForDynamicLibrary);
      return new CcLinkingInfo(
          staticModeParamsForExecutable,
          staticModeParamsForDynamicLibrary,
          dynamicModeParamsForExecutable,
          dynamicModeParamsForDynamicLibrary);
    }
  }

  @Override
  public boolean equals(Object otherObject) {
    if (!(otherObject instanceof CcLinkingInfo)) {
      return false;
    }
    CcLinkingInfo other = (CcLinkingInfo) otherObject;
    if (this == other) {
      return true;
    }
    if (!this.staticModeParamsForExecutable.equals(other.staticModeParamsForExecutable)
        || !this.staticModeParamsForDynamicLibrary.equals(other.staticModeParamsForDynamicLibrary)
        || !this.dynamicModeParamsForExecutable.equals(other.dynamicModeParamsForExecutable)
        || !this.dynamicModeParamsForDynamicLibrary.equals(
            other.dynamicModeParamsForDynamicLibrary)) {
      return false;
    }
    return true;
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(
        staticModeParamsForExecutable,
        staticModeParamsForDynamicLibrary,
        dynamicModeParamsForExecutable,
        dynamicModeParamsForDynamicLibrary);
  }
}
