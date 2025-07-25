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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StarlarkProvider.Key;
import net.starlark.java.syntax.Location;

/**
 * A struct that stores .dwo files which can be combined into a .dwp in the packaging step. See
 * https://gcc.gnu.org/wiki/DebugFission for details.
 */
@Immutable
public final class CcDebugInfoContext {
  private static final Key KEY =
      new Key(
          keyForBuiltins(Label.parseCanonicalUnchecked("@_builtins//:common/cc/cc_info.bzl")),
          "CcDebugContextInfo");
  private static final StarlarkProvider PROVIDER =
      StarlarkProvider.builder(Location.BUILTIN).buildExported(KEY);
  public static final StarlarkInfo EMPTY =
      StarlarkInfo.create(
          PROVIDER,
          ImmutableMap.of(
              "files",
              Depset.of(StarlarkInfo.class, NestedSetBuilder.emptySet(Order.STABLE_ORDER)),
              "pic_files",
              Depset.of(StarlarkInfo.class, NestedSetBuilder.emptySet(Order.STABLE_ORDER))),
          Location.BUILTIN);

  private CcDebugInfoContext() {}
}
