// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcStarlarkApiProviderApi;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * A class that exposes the C++ providers to Starlark. It is intended to provide a simple and stable
 * interface for Starlark users.
 */
public final class CcStarlarkApiInfo extends NativeInfo
    implements CcStarlarkApiProviderApi<Artifact> {
  public static final BuiltinProvider<CcStarlarkApiInfo> PROVIDER =
      new BuiltinProvider<CcStarlarkApiInfo>(
          CcStarlarkApiProvider.NAME, CcStarlarkApiInfo.class) {};

  private final CcInfo ccInfo;

  public CcStarlarkApiInfo(CcInfo ccInfo) {
    Preconditions.checkNotNull(ccInfo);
    this.ccInfo = ccInfo;
  }

  @Override
  public BuiltinProvider<CcStarlarkApiInfo> getProvider() {
    return PROVIDER;
  }

  static Depset /*<Artifact>*/ getTransitiveHeadersForStarlark(CcInfo ccInfo) {
    return Depset.of(Artifact.TYPE, getTransitiveHeaders(ccInfo));
  }

  @Override
  public Depset /*<Artifact>*/ getTransitiveHeadersForStarlark() {
    return CcStarlarkApiInfo.getTransitiveHeadersForStarlark(ccInfo);
  }

  NestedSet<Artifact> getTransitiveHeaders() {
    return CcStarlarkApiInfo.getTransitiveHeaders(ccInfo);
  }

  static NestedSet<Artifact> getTransitiveHeaders(CcInfo ccInfo) {
    CcCompilationContext ccCompilationContext = ccInfo.getCcCompilationContext();
    return ccCompilationContext.getDeclaredIncludeSrcs();
  }

  @Override
  public Depset /*<Artifact>*/ getLibrariesForStarlark() {
    return CcStarlarkApiInfo.getLibrariesForStarlark(ccInfo);
  }

  static Depset /*<Artifact>*/ getLibrariesForStarlark(CcInfo ccInfo) {
    return Depset.of(Artifact.TYPE, getLibraries(ccInfo));
  }

  NestedSet<Artifact> getLibraries() {
    return CcStarlarkApiInfo.getLibraries(ccInfo);
  }

  static NestedSet<Artifact> getLibraries(CcInfo ccInfo) {
    NestedSetBuilder<Artifact> libs = NestedSetBuilder.linkOrder();
    if (ccInfo == null) {
      return libs.build();
    }
    for (Artifact lib : ccInfo.getCcLinkingContext().getStaticModeParamsForExecutableLibraries()) {
      libs.add(lib);
    }
    return libs.build();
  }

  @Override
  public ImmutableList<String> getLinkopts() {
    return CcStarlarkApiInfo.getLinkopts(ccInfo);
  }

  static ImmutableList<String> getLinkopts(CcInfo ccInfo) {
    if (ccInfo == null) {
      return ImmutableList.of();
    }
    return ccInfo.getCcLinkingContext().getFlattenedUserLinkFlags();
  }

  @Override
  public ImmutableList<String> getDefines() {
    CcCompilationContext ccCompilationContext = ccInfo.getCcCompilationContext();
    return ccCompilationContext == null
        ? ImmutableList.<String>of()
        : ccCompilationContext.getDefines();
  }

  static ImmutableList<String> getDefines(CcInfo ccInfo) {
    CcCompilationContext ccCompilationContext = ccInfo.getCcCompilationContext();
    return ccCompilationContext == null
        ? ImmutableList.<String>of()
        : ccCompilationContext.getDefines();
  }

  @Override
  public ImmutableList<String> getSystemIncludeDirs() {
    return CcStarlarkApiInfo.getSystemIncludeDirs(ccInfo);
  }

  static ImmutableList<String> getSystemIncludeDirs(CcInfo ccInfo) {
    CcCompilationContext ccCompilationContext = ccInfo.getCcCompilationContext();
    if (ccCompilationContext == null) {
      return ImmutableList.of();
    }
    ImmutableList.Builder<String> builder = ImmutableList.builder();
    for (PathFragment path : ccCompilationContext.getSystemIncludeDirs()) {
      builder.add(path.getSafePathString());
    }
    return builder.build();
  }

  @Override
  public ImmutableList<String> getIncludeDirs() {
    return CcStarlarkApiInfo.getIncludeDirs(ccInfo);
  }

  static ImmutableList<String> getIncludeDirs(CcInfo ccInfo) {
    CcCompilationContext ccCompilationContext = ccInfo.getCcCompilationContext();
    if (ccCompilationContext == null) {
      return ImmutableList.of();
    }
    ImmutableList.Builder<String> builder = ImmutableList.builder();
    for (PathFragment path : ccCompilationContext.getIncludeDirs()) {
      builder.add(path.getSafePathString());
    }
    return builder.build();
  }

  @Override
  public ImmutableList<String> getQuoteIncludeDirs() {
    return CcStarlarkApiInfo.getQuoteIncludeDirs(ccInfo);
  }

  static ImmutableList<String> getQuoteIncludeDirs(CcInfo ccInfo) {
    CcCompilationContext ccCompilationContext = ccInfo.getCcCompilationContext();
    if (ccCompilationContext == null) {
      return ImmutableList.of();
    }
    ImmutableList.Builder<String> builder = ImmutableList.builder();
    for (PathFragment path : ccCompilationContext.getQuoteIncludeDirs()) {
      builder.add(path.getSafePathString());
    }
    return builder.build();
  }

  @Override
  public ImmutableList<String> getCcFlags() {
    return CcStarlarkApiInfo.getCcFlags(ccInfo);
  }

  static ImmutableList<String> getCcFlags(CcInfo ccInfo) {
    CcCompilationContext ccCompilationContext = ccInfo.getCcCompilationContext();

    ImmutableList.Builder<String> options = ImmutableList.builder();
    for (String define : ccCompilationContext.getDefines()) {
      options.add("-D" + define);
    }
    for (PathFragment path : ccCompilationContext.getSystemIncludeDirs()) {
      options.add("-isystem " + path.getSafePathString());
    }
    for (PathFragment path : ccCompilationContext.getIncludeDirs()) {
      options.add("-I " + path.getSafePathString());
    }
    for (PathFragment path : ccCompilationContext.getQuoteIncludeDirs()) {
      options.add("-iquote " + path.getSafePathString());
    }

    return options.build();
  }
}
