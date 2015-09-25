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
import com.google.devtools.build.lib.rules.cpp.CcLinkParams.Builder;

/**
 * A cache of C link parameters.
 *
 * <p>The cache holds instances of {@link CcLinkParams} for combinations of
 * linkingStatically and linkShared. If a requested value is not available in
 * the cache, it is computed and then stored.
 *
 * <p>Typically this class is used on targets that may be linked in as C
 * libraries as in the following example:
 *
 * <pre>
 * class SomeTarget implements CcLinkParamsProvider {
 *   private final CcLinkParamsStore ccLinkParamsStore = new CcLinkParamsStore() {
 *     @Override
 *     protected void collect(CcLinkParams.Builder builder, boolean linkingStatically,
 *                            boolean linkShared) {
 *       builder.add[...]
 *     }
 *   };
 *
 *   @Override
 *   public CcLinkParams getCcLinkParams(boolean linkingStatically, boolean linkShared) {
 *     return ccLinkParamsStore.get(linkingStatically, linkShared);
 *   }
 * }
 * </pre>
 */
public abstract class CcLinkParamsStore {

  private CcLinkParams staticSharedParams;
  private CcLinkParams staticNoSharedParams;
  private CcLinkParams noStaticSharedParams;
  private CcLinkParams noStaticNoSharedParams;

  private CcLinkParams compute(boolean linkingStatically, boolean linkShared) {
    CcLinkParams.Builder builder = CcLinkParams.builder(linkingStatically, linkShared);
    collect(builder, linkingStatically, linkShared);
    return builder.build();
  }

  /**
   * Returns {@link CcLinkParams} for a combination of parameters.
   *
   * <p>The {@link CcLinkParams} instance is computed lazily and cached.
   */
  public synchronized CcLinkParams get(boolean linkingStatically, boolean linkShared) {
    CcLinkParams result = lookup(linkingStatically, linkShared);
    if (result == null) {
      result = compute(linkingStatically, linkShared);
      put(linkingStatically, linkShared, result);
    }
    return result;
  }

  private CcLinkParams lookup(boolean linkingStatically, boolean linkShared) {
    if (linkingStatically) {
      return linkShared ? staticSharedParams : staticNoSharedParams;
    } else {
      return linkShared ? noStaticSharedParams : noStaticNoSharedParams;
    }
  }

  private void put(boolean linkingStatically, boolean linkShared, CcLinkParams params) {
    Preconditions.checkNotNull(params);
    if (linkingStatically) {
      if (linkShared) {
        staticSharedParams = params;
      } else {
        staticNoSharedParams = params;
      }
    } else {
      if (linkShared) {
        noStaticSharedParams = params;
      } else {
        noStaticNoSharedParams = params;
      }
    }
  }

  /**
   * Hook for building the actual link params.
   *
   * <p>Users should override this method and call methods of the builder to
   * set up the actual CcLinkParams objects.
   *
   * <p>Implementations of this method must not fail or try to report errors on the
   * configured target.
   */
  protected abstract void collect(CcLinkParams.Builder builder, boolean linkingStatically,
                                  boolean linkShared);

  /**
   * An empty CcLinkParamStore.
   */
  public static final CcLinkParamsStore EMPTY = new CcLinkParamsStore() {

    @Override
    protected void collect(Builder builder, boolean linkingStatically, boolean linkShared) {}
  };

  /**
   * An implementation class for the CcLinkParamsStore.
   */
  public static final class CcLinkParamsStoreImpl extends CcLinkParamsStore {

    public CcLinkParamsStoreImpl(CcLinkParamsStore store) {
      super.staticSharedParams = store.get(true, true);
      super.staticNoSharedParams = store.get(true, false);
      super.noStaticSharedParams = store.get(false, true);
      super.noStaticNoSharedParams = store.get(false, false);
    }

    @Override
    protected void collect(Builder builder, boolean linkingStatically, boolean linkShared) {}
  }
}

