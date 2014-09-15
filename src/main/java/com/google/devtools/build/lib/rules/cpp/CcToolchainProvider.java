// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.TransitiveInfoProvider;

/**
 * Information about a C++ compiler used by the <code>cc_*</code> rules.
 */
@Immutable
public final class CcToolchainProvider implements TransitiveInfoProvider {
  private final NestedSet<Artifact> crosstool;
  private final NestedSet<Artifact> crosstoolMiddleman;
  private final NestedSet<Artifact> compile;
  private final NestedSet<Artifact> strip;
  private final NestedSet<Artifact> objCopy;
  private final NestedSet<Artifact> link;
  private final NestedSet<Artifact> dwp;
  private final NestedSet<Artifact> staticRuntimeLinkInputs;
  private final Artifact staticRuntimeLinkMiddleman;
  private final NestedSet<Artifact> dynamicRuntimeLinkInputs;
  private final Artifact dynamicRuntimeLinkMiddleman;
  private final PathFragment dynamicRuntimeSolibDir;
  private final CppCompilationContext cppCompilationContext;
  private final boolean supportsParamFiles;

  public CcToolchainProvider(NestedSet<Artifact> crosstool,
      NestedSet<Artifact> crosstoolMiddleman,
      NestedSet<Artifact> compile,
      NestedSet<Artifact> strip,
      NestedSet<Artifact> objCopy,
      NestedSet<Artifact> link,
      NestedSet<Artifact> dwp,
      NestedSet<Artifact> staticRuntimeLinkInputs,
      Artifact staticRuntimeLinkMiddleman,
      NestedSet<Artifact> dynamicRuntimeLinkInputs,
      Artifact dynamicRuntimeLinkMiddleman,
      PathFragment dynamicRuntimeSolibDir,
      CppCompilationContext cppCompilationContext,
      boolean supportsParamFiles) {
    super();
    this.crosstool = crosstool;
    this.crosstoolMiddleman = crosstoolMiddleman;
    this.compile = compile;
    this.strip = strip;
    this.objCopy = objCopy;
    this.link = link;
    this.dwp = dwp;
    this.staticRuntimeLinkInputs = staticRuntimeLinkInputs;
    this.staticRuntimeLinkMiddleman = staticRuntimeLinkMiddleman;
    this.dynamicRuntimeLinkInputs = dynamicRuntimeLinkInputs;
    this.dynamicRuntimeLinkMiddleman = dynamicRuntimeLinkMiddleman;
    this.dynamicRuntimeSolibDir = dynamicRuntimeSolibDir;
    this.cppCompilationContext = cppCompilationContext;
    this.supportsParamFiles = supportsParamFiles;
  }

  /**
   * Returns all the files in Crosstool. Is not a middleman.
   */
  public NestedSet<Artifact> getCrosstool() {
    return crosstool;
  }

  /**
   * Returns a middleman for all the files in Crosstool.
   */
  public NestedSet<Artifact> getCrosstoolMiddleman() {
    return crosstoolMiddleman;
  }

  /**
   * Returns the files necessary for compilation.
   */
  public NestedSet<Artifact> getCompile() {
    return compile;
  }

  /**
   * Returns the files necessary for a 'strip' invocation.
   */
  public NestedSet<Artifact> getStrip() {
    return strip;
  }

  /**
   * Returns the files necessary for an 'objcopy' invocation.
   */
  public NestedSet<Artifact> getObjcopy() {
    return objCopy;
  }

  /**
   * Returns the files necessary for linking.
   */
  public NestedSet<Artifact> getLink() {
    return link;
  }

  public NestedSet<Artifact> getDwp() {
    return dwp;
  }

  /**
   * Returns the static runtime libraries.
   */
  public NestedSet<Artifact> getStaticRuntimeLinkInputs() {
    return staticRuntimeLinkInputs;
  }

  /**
   * Returns an aggregating middleman that represents the static runtime libraries.
   */
  public Artifact getStaticRuntimeLinkMiddleman() {
    return staticRuntimeLinkMiddleman;
  }

  /**
   * Returns the dynamic runtime libraries.
   */
  public NestedSet<Artifact> getDynamicRuntimeLinkInputs() {
    return dynamicRuntimeLinkInputs;
  }

  /**
   * Returns an aggregating middleman that represents the dynamic runtime libraries.
   */
  public Artifact getDynamicRuntimeLinkMiddleman() {
    return dynamicRuntimeLinkMiddleman;
  }

  /**
   * Returns the name of the directory where the solib symlinks for the dynamic runtime libraries
   * live. The directory itself will be under the root of the host configuration in the 'bin'
   * directory.
   */
  public PathFragment getDynamicRuntimeSolibDir() {
    return dynamicRuntimeSolibDir;
  }

  /**
   * Returns the C++ compilation context for the toolchain.
   */
  public CppCompilationContext getCppCompilationContext() {
    return cppCompilationContext;
  }

  /**
   * Whether the toolchains supports parameter files.
   */
  public boolean supportsParamFiles() {
    return supportsParamFiles;
  }
}
