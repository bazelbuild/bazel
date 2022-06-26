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
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.starlark.StarlarkApiProvider;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcStarlarkApiProviderApi;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.StarlarkValue;

/**
 * A class that exposes the C++ providers to Starlark. It is intended to provide a simple and stable
 * interface for Starlark users.
 */
@StarlarkBuiltin(
    name = "CcStarlarkApiProvider",
    category = DocCategory.PROVIDER,
    doc =
        "Provides access to information about C++ rules.  Every C++-related target provides this"
            + " struct, accessible as a <code>cc</code> field on <a"
            + " href=\"Target.html\">target</a>.")
public final class CcStarlarkApiProvider extends StarlarkApiProvider
    implements CcStarlarkApiProviderApi<Artifact>, StarlarkValue {
  /** The name of the field in Starlark used to access this class. */
  public static final String NAME = "cc";

  public static void maybeAdd(RuleContext ruleContext, RuleConfiguredTargetBuilder builder) {
    if (ruleContext.getFragment(CppConfiguration.class).enableLegacyCcProvider()) {
      builder.addStarlarkTransitiveInfo(NAME, new CcStarlarkApiProvider());
    }
  }

  @Override
  public Depset /*<Artifact>*/ getTransitiveHeadersForStarlark() {
    return CcStarlarkApiInfo.getTransitiveHeadersForStarlark(getInfo().get(CcInfo.PROVIDER));
  }

  NestedSet<Artifact> getTransitiveHeaders() {
    return CcStarlarkApiInfo.getTransitiveHeaders(getInfo().get(CcInfo.PROVIDER));
  }

  @Override
  public Depset /*<Artifact>*/ getLibrariesForStarlark() {
    return CcStarlarkApiInfo.getLibrariesForStarlark(getInfo().get(CcInfo.PROVIDER));
  }

  NestedSet<Artifact> getLibraries() {
    return CcStarlarkApiInfo.getLibraries(getInfo().get(CcInfo.PROVIDER));
  }

  @Override
  public ImmutableList<String> getLinkopts() {
    return CcStarlarkApiInfo.getLinkopts(getInfo().get(CcInfo.PROVIDER));
  }

  @Override
  public ImmutableList<String> getDefines() {
    CcCompilationContext ccCompilationContext =
        getInfo().get(CcInfo.PROVIDER).getCcCompilationContext();
    return ccCompilationContext == null
        ? ImmutableList.<String>of()
        : ccCompilationContext.getDefines();
  }

  @Override
  public ImmutableList<String> getSystemIncludeDirs() {
    return CcStarlarkApiInfo.getSystemIncludeDirs(getInfo().get(CcInfo.PROVIDER));
  }

  @Override
  public ImmutableList<String> getIncludeDirs() {
    return CcStarlarkApiInfo.getIncludeDirs(getInfo().get(CcInfo.PROVIDER));
  }

  @Override
  public ImmutableList<String> getQuoteIncludeDirs() {
    return CcStarlarkApiInfo.getQuoteIncludeDirs(getInfo().get(CcInfo.PROVIDER));
  }

  @Override
  public ImmutableList<String> getCcFlags() {
    return CcStarlarkApiInfo.getCcFlags(getInfo().get(CcInfo.PROVIDER));
  }
}
