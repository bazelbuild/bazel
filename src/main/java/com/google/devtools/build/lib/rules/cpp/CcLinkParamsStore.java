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

import com.google.common.base.Function;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;

/** An implementation class for the AbstractCcLinkParamsStore. */
@AutoCodec
@Deprecated
// TODO(plf): Remove class, use CcLinkParams instances individually.
public final class CcLinkParamsStore extends AbstractCcLinkParamsStore {
  public static final ObjectCodec<com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore> CODEC =
      new CcLinkParamsStore_AutoCodec();
  public static final Function<TransitiveInfoCollection, CcLinkingInfo> TO_LINK_PARAMS =
      input -> {
        // ... then try Skylark.
        return input.get(CcLinkingInfo.PROVIDER);
      };

  @AutoCodec
  @VisibleForSerialization
  static class CcLinkParamsInfoCollection extends AbstractCcLinkParamsStore {
    private final Iterable<CcLinkingInfo> ccLinkingInfos;

    CcLinkParamsInfoCollection(Iterable<CcLinkingInfo> ccLinkingInfos) {
      this.ccLinkingInfos = ccLinkingInfos;
    }

    @Override
    protected void collect(
        CcLinkParams.Builder builder, boolean linkingStatically, boolean linkShared) {
      for (CcLinkingInfo ccLinkingInfo : ccLinkingInfos) {
        builder.add(ccLinkingInfo);
      }
    }
  }

  public CcLinkParamsStore(AbstractCcLinkParamsStore store) {
    this(
        store.get(true, true),
        store.get(true, false),
        store.get(false, true),
        store.get(false, false));
  }

  @VisibleForSerialization
  @AutoCodec.Instantiator
  public CcLinkParamsStore(
      CcLinkParams staticModeParamsForDynamicLibrary,
      CcLinkParams staticModeParamsForExecutable,
      CcLinkParams dynamicModeParamsForDynamicLibrary,
      CcLinkParams dynamicModeParamsForExecutable) {
    super.staticModeParamsForDynamicLibrary = staticModeParamsForDynamicLibrary;
    super.staticModeParamsForExecutable = staticModeParamsForExecutable;
    super.dynamicModeParamsForDynamicLibrary = dynamicModeParamsForDynamicLibrary;
    super.dynamicModeParamsForExecutable = dynamicModeParamsForExecutable;
  }

  public static com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore merge(
      final Iterable<CcLinkingInfo> providers) {
    return new com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore(
        new CcLinkParamsInfoCollection(providers));
  }

  @Override
  protected void collect(
      CcLinkParams.Builder builder, boolean linkingStatically, boolean linkShared) {}

  /** Returns link parameters given static / shared linking settings. */
  public CcLinkParams getCcLinkParams(boolean linkingStatically, boolean linkShared) {
    return get(linkingStatically, linkShared);
  }
}
