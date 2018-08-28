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
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcLinkingInfoApi;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkType;
import java.util.Collection;
import javax.annotation.Nullable;

/** Wrapper for every C++ linking provider. */
@Immutable
@AutoCodec
public final class CcLinkingInfo extends NativeInfo implements CcLinkingInfoApi {

  private static final FunctionSignature.WithValues<Object, SkylarkType> SIGNATURE =
      FunctionSignature.WithValues.create(
          FunctionSignature.of(
              /* numMandatoryPositionals= */ 0,
              /* numOptionalPositionals= */ 0,
              // TODO(plf): Make CcLinkParams parameters mandatory once existing rules have been
              // migrated.
              /* numMandatoryNamedOnly= */ 0,
              /* starArg= */ false,
              /* kwArg= */ false,
              "static_mode_params_for_dynamic_library",
              "static_mode_params_for_executable",
              "dynamic_mode_params_for_dynamic_library",
              "dynamic_mode_params_for_executable",
              "cc_runfiles"),
          /* defaultValues= */ ImmutableList.of(
              Runtime.NONE, Runtime.NONE, Runtime.NONE, Runtime.NONE, Runtime.NONE),
          /* types= */ ImmutableList.of(
              SkylarkType.of(CcLinkParams.class),
              SkylarkType.of(CcLinkParams.class),
              SkylarkType.of(CcLinkParams.class),
              SkylarkType.of(CcLinkParams.class),
              SkylarkType.of(CcRunfiles.class)));

  @Nullable
  private static Object nullIfNone(Object object) {
    return nullIfNone(object, Object.class);
  }

  @Nullable
  private static <T> T nullIfNone(Object object, Class<T> type) {
    return object != Runtime.NONE ? type.cast(object) : null;
  }

  public static final NativeProvider<CcLinkingInfo> PROVIDER =
      new NativeProvider<CcLinkingInfo>(CcLinkingInfo.class, "CcLinkingInfo", SIGNATURE) {
        @Override
        @SuppressWarnings("unchecked")
        protected CcLinkingInfo createInstanceFromSkylark(
            Object[] args, Environment env, Location loc) throws EvalException {
          CcCommon.checkLocationWhitelisted(
              env.getSemantics(),
              loc,
              env.getGlobals().getTransitiveLabel().getPackageIdentifier().toString());
          int i = 0;
          CcLinkParams staticModeParamsForDynamicLibrary = (CcLinkParams) nullIfNone(args[i++]);
          CcLinkParams staticModeParamsForExecutable = (CcLinkParams) nullIfNone(args[i++]);
          CcLinkParams dynamicModeParamsForDynamicLibrary = (CcLinkParams) nullIfNone(args[i++]);
          CcLinkParams dynamicModeParamsForExecutable = (CcLinkParams) nullIfNone(args[i++]);
          CcRunfiles ccRunfiles = (CcRunfiles) nullIfNone(args[i++]);
          CcLinkingInfo.Builder ccLinkingInfoBuilder = CcLinkingInfo.Builder.create();
          if (staticModeParamsForDynamicLibrary == null
              || staticModeParamsForExecutable == null
              || dynamicModeParamsForDynamicLibrary == null
              || dynamicModeParamsForExecutable == null) {
            throw new EvalException(
                loc, "Every CcLinkParams parameter must be passed to CcLinkingInfo.");
          }
          ccLinkingInfoBuilder
              .setStaticModeParamsForDynamicLibrary(staticModeParamsForDynamicLibrary)
              .setStaticModeParamsForExecutable(staticModeParamsForExecutable)
              .setDynamicModeParamsForDynamicLibrary(dynamicModeParamsForDynamicLibrary)
              .setDynamicModeParamsForExecutable(dynamicModeParamsForExecutable);
          // TODO(plf): The CcDynamicLibrariesForRuntime provider can be removed perhaps. Do not
          // add to the API until we know for sure. The CcRunfiles provider is already in the API
          // at the time of this comment (cl/200184914). Perhaps it can be removed but Skylark rules
          // using it will have to be migrated.
          ccLinkingInfoBuilder.setCcRunfiles(ccRunfiles);
          return ccLinkingInfoBuilder.build();
        }
      };

  public static final CcLinkingInfo EMPTY =
      CcLinkingInfo.Builder.create()
          .setStaticModeParamsForDynamicLibrary(CcLinkParams.EMPTY)
          .setStaticModeParamsForExecutable(CcLinkParams.EMPTY)
          .setDynamicModeParamsForDynamicLibrary(CcLinkParams.EMPTY)
          .setDynamicModeParamsForExecutable(CcLinkParams.EMPTY)
          .build();

  private final CcLinkParamsStore ccLinkParamsStore;
  // TODO(b/111289526): These providers are not useful. All the information they provide can be
  // obtained from CcLinkParams. CcRunfiles is already dead code and can be removed.
  // CcDynamicLibrariesForRuntime is not dead code yet.
  @Deprecated private final CcRunfiles ccRunfiles;
  @Deprecated private final CcDynamicLibrariesForRuntime ccDynamicLibrariesForRuntime;

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
  public CcLinkParams getStaticModeParamsForDynamicLibrary() {
    return ccLinkParamsStore.get(/* linkingStatically= */ true, /* linkShared= */ true);
  }

  @Override
  public CcLinkParams getStaticModeParamsForExecutable() {
    return ccLinkParamsStore.get(/* linkingStatically= */ true, /* linkShared= */ false);
  }

  @Override
  public CcLinkParams getDynamicModeParamsForDynamicLibrary() {
    return ccLinkParamsStore.get(/* linkingStatically= */ false, /* linkShared= */ true);
  }

  @Override
  public CcLinkParams getDynamicModeParamsForExecutable() {
    return ccLinkParamsStore.get(/* linkingStatically= */ false, /* linkShared= */ false);
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

  @Override
  public CcRunfiles getCcRunfiles() {
    return ccRunfiles;
  }

  public CcDynamicLibrariesForRuntime getCcDynamicLibrariesForRuntime() {
    return ccDynamicLibrariesForRuntime;
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
    CcRunfiles ccRunfiles;
    CcDynamicLibrariesForRuntime ccDynamicLibrariesForRuntime;

    public static CcLinkingInfo.Builder create() {
      return new CcLinkingInfo.Builder();
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
      Preconditions.checkNotNull(staticModeParamsForDynamicLibrary);
      Preconditions.checkNotNull(staticModeParamsForExecutable);
      Preconditions.checkNotNull(dynamicModeParamsForDynamicLibrary);
      Preconditions.checkNotNull(dynamicModeParamsForExecutable);
      CcLinkParamsStore ccLinkParamsStore =
          new CcLinkParamsStore(
              staticModeParamsForDynamicLibrary,
              staticModeParamsForExecutable,
              dynamicModeParamsForDynamicLibrary,
              dynamicModeParamsForExecutable);
      return new CcLinkingInfo(ccLinkParamsStore, ccRunfiles, ccDynamicLibrariesForRuntime);
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
    if (!this.ccLinkParamsStore.equals(other.ccLinkParamsStore)) {
      return false;
    }
    return true;
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(ccLinkParamsStore);
  }
}
