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
public final class CcLinkParamsStore extends AbstractCcLinkParamsStore {
  public static final ObjectCodec<com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore> CODEC =
      new CcLinkParamsStore_AutoCodec();
  public static final Function<TransitiveInfoCollection, AbstractCcLinkParamsStore> TO_LINK_PARAMS =
      input -> {
        // ... then try Skylark.
        CcLinkingInfo provider = input.get(CcLinkingInfo.PROVIDER);
        return provider == null ? null : provider.getCcLinkParamsStore();
      };

  @AutoCodec
  @VisibleForSerialization
  static class CcLinkParamsInfoCollection extends AbstractCcLinkParamsStore {
    private final Iterable<com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore>
        ccLinkParamStores;

    CcLinkParamsInfoCollection(
        Iterable<com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore> ccLinkParamStores) {
      this.ccLinkParamStores = ccLinkParamStores;
    }

    @Override
    protected void collect(
        CcLinkParams.Builder builder, boolean linkingStatically, boolean linkShared) {
      for (com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore ccLinkParamsStore :
          ccLinkParamStores) {
        builder.add(ccLinkParamsStore);
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
      CcLinkParams staticSharedParams,
      CcLinkParams staticNoSharedParams,
      CcLinkParams noStaticSharedParams,
      CcLinkParams noStaticNoSharedParams) {
    super.staticSharedParams = staticSharedParams;
    super.staticNoSharedParams = staticNoSharedParams;
    super.noStaticSharedParams = noStaticSharedParams;
    super.noStaticNoSharedParams = noStaticNoSharedParams;
  }

  public static com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore merge(
      final Iterable<com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore> providers) {
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
