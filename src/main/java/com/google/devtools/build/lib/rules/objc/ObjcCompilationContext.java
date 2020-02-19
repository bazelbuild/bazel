// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.HEADER;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.cpp.CcCompilationContext;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.List;

/**
 * A collection of compilation information gathered for a particular rule. This is used to generate
 * the compilation command line and to the supply information that goes into the compilation info
 * provider.
 */
@Immutable
public final class ObjcCompilationContext {
  public static final ObjcCompilationContext EMPTY =
      builder(StarlarkSemantics.DEFAULT_SEMANTICS).build();

  private final ImmutableList<String> defines;

  /**
   * The list of public headers. We expect this to contain both the headers from the src attribute,
   * as well as any "additional" headers required for compilation.
   */
  private final ImmutableList<Artifact> publicHeaders;

  private final ImmutableList<Artifact> publicTextualHeaders;
  private final ImmutableList<Artifact> privateHeaders;
  private final ImmutableList<PathFragment> includes;
  private final ImmutableList<PathFragment> systemIncludes;
  private final ImmutableList<PathFragment> quoteIncludes;
  private final ImmutableList<ObjcProvider> depObjcProviders;
  private final ImmutableList<CcCompilationContext> depCcCompilationContexts;
  private final ObjcProvider depObjcProvidersSummary;

  ObjcCompilationContext(
      Iterable<String> defines,
      Iterable<Artifact> publicHeaders,
      Iterable<Artifact> publicTextualHeaders,
      Iterable<Artifact> privateHeaders,
      Iterable<PathFragment> includes,
      Iterable<PathFragment> systemIncludes,
      Iterable<PathFragment> quoteIncludes,
      Iterable<ObjcProvider> depObjcProviders,
      Iterable<CcCompilationContext> depCcCompilationContexts,
      ObjcProvider depObjcProvidersSummary) {
    this.defines = ImmutableList.copyOf(defines);
    this.publicHeaders = ImmutableList.copyOf(publicHeaders);
    this.publicTextualHeaders = ImmutableList.copyOf(publicTextualHeaders);
    this.privateHeaders = ImmutableList.copyOf(privateHeaders);
    this.includes = ImmutableList.copyOf(includes);
    this.systemIncludes = ImmutableList.copyOf(systemIncludes);
    this.quoteIncludes = ImmutableList.copyOf(quoteIncludes);
    this.depObjcProviders = ImmutableList.copyOf(depObjcProviders);
    this.depCcCompilationContexts = ImmutableList.copyOf(depCcCompilationContexts);
    this.depObjcProvidersSummary = depObjcProvidersSummary;
  }

  public ImmutableList<String> getDefines() {
    return defines;
  }

  public ImmutableList<Artifact> getPublicHeaders() {
    return publicHeaders;
  }

  public ImmutableList<Artifact> getPublicTextualHeaders() {
    return publicTextualHeaders;
  }

  public ImmutableList<Artifact> getPrivateHeaders() {
    return privateHeaders;
  }

  public ImmutableList<PathFragment> getIncludes() {
    return includes;
  }

  public ImmutableList<PathFragment> getSystemIncludes() {
    return systemIncludes;
  }

  public ImmutableList<PathFragment> getQuoteIncludes() {
    return quoteIncludes;
  }

  public ImmutableList<ObjcProvider> getDepObjcProviders() {
    return depObjcProviders;
  }

  public ImmutableList<CcCompilationContext> getDepCcCompilationContexts() {
    return depCcCompilationContexts;
  }

  public ObjcProvider getDepObjcProvidersSummary() {
    return depObjcProvidersSummary;
  }

  // Note: this function will be deleted in the migration cl.
  public ImmutableList<Artifact> getAllTransitiveHeaders() {
    ImmutableList.Builder<Artifact> allHeaders = ImmutableList.builder();
    allHeaders.addAll(publicHeaders);
    allHeaders.addAll(publicTextualHeaders);
    allHeaders.addAll(depObjcProvidersSummary.get(HEADER).toList());
    for (CcCompilationContext ccCompilationContext : depCcCompilationContexts) {
      allHeaders.addAll(ccCompilationContext.getDeclaredIncludeSrcs().toList());
    }
    return allHeaders.build();
  }

  public static Builder builder(StarlarkSemantics semantics) {
    return new Builder(semantics);
  }

  static class Builder {
    private final StarlarkSemantics semantics;
    private final List<String> defines = new ArrayList<>();
    private final List<Artifact> publicHeaders = new ArrayList<>();
    private final List<Artifact> publicTextualHeaders = new ArrayList<>();
    private final List<Artifact> privateHeaders = new ArrayList<>();
    private final List<PathFragment> includes = new ArrayList<>();
    private final List<PathFragment> systemIncludes = new ArrayList<>();
    private final List<PathFragment> quoteIncludes = new ArrayList<>();
    private final List<ObjcProvider> depObjcProviders = new ArrayList<>();
    private final List<CcCompilationContext> depCcCompilationContexts = new ArrayList<>();

    Builder(StarlarkSemantics semantics) {
      this.semantics = semantics;
    }

    public Builder addDefines(Iterable<String> defines) {
      Iterables.addAll(this.defines, defines);
      return this;
    }

    public Builder addPublicHeaders(Iterable<Artifact> headers) {
      Iterables.addAll(this.publicHeaders, headers);
      return this;
    }

    public Builder addPublicTextualHeaders(Iterable<Artifact> headers) {
      Iterables.addAll(this.publicTextualHeaders, headers);
      return this;
    }

    public Builder addPrivateHeaders(Iterable<Artifact> headers) {
      Iterables.addAll(this.privateHeaders, headers);
      return this;
    }

    public Builder addIncludes(Iterable<PathFragment> includes) {
      Iterables.addAll(this.includes, includes);
      return this;
    }

    public Builder addSystemIncludes(Iterable<PathFragment> includes) {
      Iterables.addAll(this.systemIncludes, includes);
      return this;
    }

    public Builder addQuoteIncludes(Iterable<PathFragment> includes) {
      Iterables.addAll(this.quoteIncludes, includes);
      return this;
    }

    public Builder addDepObjcProviders(Iterable<ObjcProvider> objcProviders) {
      // After migration, we will have nice embedded CcCompilationContexts that we can
      // addDepCcCompilationContexts().
      Iterables.addAll(this.depObjcProviders, objcProviders);
      return this;
    }

    public Builder addDepCcCompilationContexts(
        Iterable<CcCompilationContext> ccCompilationContexts) {
      Iterables.addAll(this.depCcCompilationContexts, ccCompilationContexts);
      return this;
    }

    public Builder addDepCcCompilationContext(CcCompilationContext ccCompilationContext) {
      this.depCcCompilationContexts.add(ccCompilationContext);
      return this;
    }

    ObjcCompilationContext build() {
      ObjcProvider.Builder depObjcProvidersSummaryBuilder = new ObjcProvider.Builder(semantics);
      for (ObjcProvider dep : depObjcProviders) {
        depObjcProvidersSummaryBuilder.addTransitiveAndPropagateCompileInfo(dep);
      }
      ObjcProvider depObjcProvidersSummary = depObjcProvidersSummaryBuilder.build();

      return new ObjcCompilationContext(
          defines,
          publicHeaders,
          publicTextualHeaders,
          privateHeaders,
          includes,
          systemIncludes,
          quoteIncludes,
          depObjcProviders,
          depCcCompilationContexts,
          depObjcProvidersSummary);
    }
  }
}
