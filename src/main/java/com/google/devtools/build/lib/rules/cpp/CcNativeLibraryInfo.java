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

import static com.google.devtools.build.lib.skyframe.BzlLoadValue.keyForBuiltins;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import java.util.List;
import net.starlark.java.eval.EvalException;
import net.starlark.java.syntax.Location;

/** Unwraps information about C++ libraries. */
@Immutable
public final class CcNativeLibraryInfo {
  private static final StarlarkProvider.Key KEY =
      new StarlarkProvider.Key(
          keyForBuiltins(Label.parseCanonicalUnchecked("@_builtins//:common/cc/cc_info.bzl")),
          "CcNativeLibraryInfo");
  private static final StarlarkProvider PROVIDER =
      StarlarkProvider.builder(Location.BUILTIN).buildExported(KEY);

  private final StarlarkInfo value;

  public static final StarlarkInfo EMPTY = of(NestedSetBuilder.emptySet(Order.LINK_ORDER));

  private CcNativeLibraryInfo(StarlarkInfo value) {
    this.value = value;
  }

  public static StarlarkInfo of(NestedSet<StarlarkInfo> transitiveCcNativeLibraries) {
    return StarlarkInfo.create(
        PROVIDER,
        ImmutableMap.of(
            "libraries_to_link", Depset.of(StarlarkInfo.class, transitiveCcNativeLibraries)),
        Location.BUILTIN);
  }

  /**
   * @deprecated Use only in tests
   */
  @Deprecated
  public static CcNativeLibraryInfo wrap(StarlarkInfo ccNativeLibraryInfo) {
    return new CcNativeLibraryInfo(ccNativeLibraryInfo);
  }

  /**
   * Collects native libraries in the transitive closure of its deps that are needed for executing
   * C/C++ code.
   *
   * <p>In effect, returns all dynamic library (.so) artifacts provided by the transitive closure.
   */
  public static NestedSet<StarlarkInfo> getTransitiveCcNativeLibraries(
      StarlarkInfo ccNativeLibraryInfo) {
    try {
      return Depset.cast(
          ccNativeLibraryInfo.getValue("libraries_to_link", Depset.class),
          StarlarkInfo.class,
          "libraries_to_link");
    } catch (EvalException e) {
      // Can't happen
      throw new IllegalArgumentException(e);
    }
  }

  /**
   * @deprecated Use only in tests
   */
  @Deprecated
  public NestedSet<LibraryToLink> getTransitiveCcNativeLibrariesForTests() {
    return LibraryToLink.wrap(getTransitiveCcNativeLibraries(value));
  }

  /** Merge several CcNativeLibraryInfo objects into one. */
  public static StarlarkInfo merge(List<StarlarkInfo> providers) {
    if (providers.isEmpty()) {
      return EMPTY;
    } else if (providers.size() == 1) {
      return Iterables.getOnlyElement(providers);
    }

    // TODO(b/425863238): Test the order of CCInfo.libraries_to_link
    NestedSetBuilder<StarlarkInfo> transitiveCcNativeLibraries = NestedSetBuilder.linkOrder();
    for (StarlarkInfo provider : providers) {
      transitiveCcNativeLibraries.addTransitive(getTransitiveCcNativeLibraries(provider));
    }
    return CcNativeLibraryInfo.of(transitiveCcNativeLibraries.build());
  }
}
