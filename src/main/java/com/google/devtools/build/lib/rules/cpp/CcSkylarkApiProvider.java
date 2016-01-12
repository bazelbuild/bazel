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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.SkylarkApiProvider;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * A class that exposes the C++ providers to Skylark. It is intended to provide a
 * simple and stable interface for Skylark users.
 */
@SkylarkModule(
    name = "CcSkylarkApiProvider", doc = "Provides access to information about C++ rules")
public final class CcSkylarkApiProvider extends SkylarkApiProvider {
  /** The name of the field in Skylark used to access this class. */
  public static final String NAME = "cc";

  @SkylarkCallable(
      name = "transitive_headers",
      structField = true,
      doc =
          "Returns the immutable set of headers that have been declared in the <code>src</code> "
              + "or <code>headers</code> attribute (possibly empty but never None).")
  public NestedSet<Artifact> getTransitiveHeaders() {
    CppCompilationContext ccContext = getInfo().getProvider(CppCompilationContext.class);
    return ccContext.getDeclaredIncludeSrcs();
  }

  @SkylarkCallable(
      name = "libs",
      structField = true,
      doc =
          "Returns the immutable set of libraries for either "
              + "FULLY STATIC mode (linkopts=[\"-static\"]) or MOSTLY STATIC mode (linkstatic=1) "
              + "(possibly empty but never None)")
  public NestedSet<Artifact> getLibraries() {
    NestedSetBuilder<Artifact> libs = NestedSetBuilder.linkOrder();
    CcLinkParamsProvider ccLinkParams = getInfo().getProvider(CcLinkParamsProvider.class);
    if (ccLinkParams == null) {
      return libs.build();
    }
    for (LinkerInput lib : ccLinkParams.getCcLinkParams(true, false).getLibraries()) {
      libs.add(lib.getArtifact());
    }
    return libs.build();
  }

  @SkylarkCallable(
      name = "link_flags",
      structField = true,
      doc =
          "Returns the immutable list of flags given to the C++ linker command for either "
              + "FULLY STATIC mode (linkopts=[\"-static\"]) or MOSTLY STATIC mode (linkstatic=1) "
              + "(possibly empty but never None)")
  public ImmutableList<String> getLinkopts() {
    CcLinkParamsProvider ccLinkParams = getInfo().getProvider(CcLinkParamsProvider.class);
    if (ccLinkParams == null) {
      return ImmutableList.of();
    }
    return ccLinkParams.getCcLinkParams(true, false).flattenedLinkopts();
  }

  @SkylarkCallable(
      name = "compile_flags",
      structField = true,
      doc =
          "Returns the immutable set of flags used to compile this target "
              + "(possibly empty but never None).")
  public ImmutableList<String> getCcFlags() {
    CppCompilationContext ccContext = getInfo().getProvider(CppCompilationContext.class);

    ImmutableList.Builder<String> options = ImmutableList.builder();
    for (String define : ccContext.getDefines()) {
      options.add("-D" + define);
    }
    for (PathFragment path : ccContext.getSystemIncludeDirs()) {
      options.add("-isystem " + path.getSafePathString());
    }
    for (PathFragment path : ccContext.getIncludeDirs()) {
      options.add("-I " + path.getSafePathString());
    }
    for (PathFragment path : ccContext.getQuoteIncludeDirs()) {
      options.add("-iquote " + path.getSafePathString());
    }

    return options.build();
  }
}
