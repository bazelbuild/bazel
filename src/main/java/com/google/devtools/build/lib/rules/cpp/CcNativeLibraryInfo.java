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

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcNativeLibraryInfoApi;
import java.util.List;
import net.starlark.java.eval.EvalException;

/**
 * A target that provides native libraries in the transitive closure of its deps that are needed for
 * executing C++ code.
 */
@Immutable
public final class CcNativeLibraryInfo extends NativeInfo implements CcNativeLibraryInfoApi {

  public static final CcNativeLibraryInfo EMPTY =
      new CcNativeLibraryInfo(NestedSetBuilder.emptySet(Order.LINK_ORDER));

  private final NestedSet<LibraryToLink> transitiveCcNativeLibraries;

  public static final Provider PROVIDER = new Provider();

  public CcNativeLibraryInfo(NestedSet<LibraryToLink> transitiveCcNativeLibraries) {
    this.transitiveCcNativeLibraries = transitiveCcNativeLibraries;
  }

  @Override
  public Provider getProvider() {
    return PROVIDER;
  }

  /**
   * Collects native libraries in the transitive closure of its deps that are needed for executing
   * C/C++ code.
   *
   * <p>In effect, returns all dynamic library (.so) artifacts provided by the transitive closure.
   */
  public NestedSet<LibraryToLink> getTransitiveCcNativeLibraries() {
    return transitiveCcNativeLibraries;
  }

  /** Merge several CcNativeLibraryInfo objects into one. */
  public static CcNativeLibraryInfo merge(List<CcNativeLibraryInfo> providers) {
    if (providers.isEmpty()) {
      return EMPTY;
    } else if (providers.size() == 1) {
      return Iterables.getOnlyElement(providers);
    }

    NestedSetBuilder<LibraryToLink> transitiveCcNativeLibraries = NestedSetBuilder.linkOrder();
    for (CcNativeLibraryInfo provider : providers) {
      transitiveCcNativeLibraries.addTransitive(provider.getTransitiveCcNativeLibraries());
    }
    return new CcNativeLibraryInfo(transitiveCcNativeLibraries.build());
  }

  /** Provider class for {@link CcNativeLibraryInfo} objects */
  public static class Provider extends BuiltinProvider<CcNativeLibraryInfo>
      implements CcNativeLibraryInfoApi.Provider {
    private Provider() {
      super(CcNativeLibraryInfoApi.NAME, CcNativeLibraryInfo.class);
    }

    @Override
    public CcNativeLibraryInfo createCcNativeLibraryInfo(Object librariesToLinkObject)
        throws EvalException {
      NestedSet<LibraryToLink> librariesToLink =
          Depset.cast(librariesToLinkObject, LibraryToLink.class, "libraries_to_link");
      if (librariesToLink.isEmpty()) {
        return EMPTY;
      }
      return new CcNativeLibraryInfo(librariesToLink);
    }
  }
}
