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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.StarlarkInfoWithSchema;
import com.google.devtools.build.lib.rules.cpp.CcCompilationContext;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.ArrayList;
import java.util.List;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkValue;

/**
 * A collection of compilation information gathered for a particular rule. This is used to generate
 * the compilation command line and to the supply information that goes into the compilation info
 * provider.
 */
@Immutable
public final class ObjcCompilationContext implements StarlarkValue {
  public static final ObjcCompilationContext EMPTY = builder().build();

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
  private final ImmutableList<PathFragment> strictDependencyIncludes;
  private final ImmutableList<CcCompilationContext> directCcCompilationContexts;
  private final ImmutableList<CcCompilationContext> ccCompilationContexts;
  private final ImmutableList<CcCompilationContext> implementationCcCompilationContexts;

  ObjcCompilationContext(
      Iterable<String> defines,
      Iterable<Artifact> publicHeaders,
      Iterable<Artifact> publicTextualHeaders,
      Iterable<Artifact> privateHeaders,
      Iterable<PathFragment> includes,
      Iterable<PathFragment> systemIncludes,
      Iterable<PathFragment> quoteIncludes,
      Iterable<PathFragment> strictDependencyIncludes,
      Iterable<CcCompilationContext> directCcCompilationContexts,
      Iterable<CcCompilationContext> ccCompilationContexts,
      Iterable<CcCompilationContext> implementationCcCompilationContexts) {
    this.defines = ImmutableList.copyOf(defines);
    this.publicHeaders = ImmutableList.copyOf(publicHeaders);
    this.publicTextualHeaders = ImmutableList.copyOf(publicTextualHeaders);
    this.privateHeaders = ImmutableList.copyOf(privateHeaders);
    this.includes = ImmutableList.copyOf(includes);
    this.systemIncludes = ImmutableList.copyOf(systemIncludes);
    this.quoteIncludes = ImmutableList.copyOf(quoteIncludes);
    this.strictDependencyIncludes = ImmutableList.copyOf(strictDependencyIncludes);
    this.directCcCompilationContexts = ImmutableList.copyOf(directCcCompilationContexts);
    this.ccCompilationContexts = ImmutableList.copyOf(ccCompilationContexts);
    this.implementationCcCompilationContexts =
        ImmutableList.copyOf(implementationCcCompilationContexts);
  }

  public ImmutableList<String> getDefines() {
    return defines;
  }

  @StarlarkMethod(name = "defines", documented = false, structField = true)
  public Sequence<String> getDefinesForStarlark() {
    return StarlarkList.immutableCopyOf(getDefines());
  }

  public ImmutableList<Artifact> getPublicHeaders() {
    return publicHeaders;
  }

  public ImmutableList<Artifact> getPublicTextualHeaders() {
    return publicTextualHeaders;
  }

  @StarlarkMethod(name = "public_textual_hdrs", documented = false, structField = true)
  public Sequence<Artifact> getPublicTextualHeadersForStarlark() {
    return StarlarkList.immutableCopyOf(getPublicTextualHeaders());
  }

  public ImmutableList<Artifact> getPrivateHeaders() {
    return privateHeaders;
  }

  public ImmutableList<PathFragment> getIncludes() {
    return includes;
  }

  @StarlarkMethod(name = "includes", documented = false, structField = true)
  public Sequence<String> getIncludesForStarlark() {
    return StarlarkList.immutableCopyOf(
        getIncludes().stream()
            .map(PathFragment::getSafePathString)
            .collect(ImmutableList.toImmutableList()));
  }

  public ImmutableList<PathFragment> getSystemIncludes() {
    return systemIncludes;
  }

  @StarlarkMethod(name = "system_includes", documented = false, structField = true)
  public Sequence<String> getSystemIncludesForStarlark() {
    return StarlarkList.immutableCopyOf(
        getSystemIncludes().stream()
            .map(PathFragment::getSafePathString)
            .collect(ImmutableList.toImmutableList()));
  }

  public ImmutableList<PathFragment> getQuoteIncludes() {
    return quoteIncludes;
  }

  @StarlarkMethod(name = "quote_includes", documented = false, structField = true)
  public Sequence<String> getQuoteIncludesForStarlark() {
    return StarlarkList.immutableCopyOf(
        getQuoteIncludes().stream()
            .map(PathFragment::getSafePathString)
            .collect(ImmutableList.toImmutableList()));
  }

  public ImmutableList<PathFragment> getStrictDependencyIncludes() {
    return strictDependencyIncludes;
  }

  @StarlarkMethod(name = "strict_dependency_includes", documented = false, structField = true)
  public Sequence<String> getStrictDependencyIncludesForStarlark() {
    return StarlarkList.immutableCopyOf(
        getStrictDependencyIncludes().stream()
            .map(PathFragment::getSafePathString)
            .collect(ImmutableList.toImmutableList()));
  }

  public ImmutableList<CcCompilationContext> getDirectCcCompilationContexts() {
    return directCcCompilationContexts;
  }

  public ImmutableList<CcCompilationContext> getCcCompilationContexts() {
    return ccCompilationContexts;
  }

  public ImmutableList<CcCompilationContext> getImplementationCcCompilationContexts() {
    return implementationCcCompilationContexts;
  }

  @StarlarkMethod(name = "cc_compilation_contexts", documented = false, structField = true)
  public Sequence<CcCompilationContext> getCcCompilationContextsForStarlark() {
    return StarlarkList.immutableCopyOf(getCcCompilationContexts());
  }

  @StarlarkMethod(
      name = "implementation_cc_compilation_contexts",
      documented = false,
      structField = true)
  public Sequence<CcCompilationContext> getImplementationCcCompilationContextsForStarlark() {
    return StarlarkList.immutableCopyOf(getImplementationCcCompilationContexts());
  }

  @StarlarkMethod(name = "create_cc_compilation_context", documented = false)
  public CcCompilationContext createCcCompilationContext() {
    CcCompilationContext.Builder builder = CcCompilationContext.builder();
    builder
        .addDefines(getDefines())
        .addDeclaredIncludeSrcs(getPublicHeaders())
        .addDeclaredIncludeSrcs(getPrivateHeaders())
        .addDeclaredIncludeSrcs(getPublicTextualHeaders())
        .addModularPublicHdrs(ImmutableList.copyOf(getPublicHeaders()))
        .addModularPrivateHdrs(ImmutableList.copyOf(getPrivateHeaders()))
        .addTextualHdrs(ImmutableList.copyOf(getPublicTextualHeaders()))
        .addIncludeDirs(getIncludes())
        .addSystemIncludeDirs(getSystemIncludes())
        .addQuoteIncludeDirs(getQuoteIncludes())
        .addDependentCcCompilationContexts(
            getDirectCcCompilationContexts(), getCcCompilationContexts());
    return builder.build();
  }

  /** Create and return an initial empty Builder for ObjcCompilationContext. */
  public static Builder builder() {
    return new Builder();
  }

  static class Builder {
    private final List<String> defines = new ArrayList<>();
    private final List<Artifact> publicHeaders = new ArrayList<>();
    private final List<Artifact> publicTextualHeaders = new ArrayList<>();
    private final List<Artifact> privateHeaders = new ArrayList<>();
    private final List<PathFragment> includes = new ArrayList<>();
    private final List<PathFragment> systemIncludes = new ArrayList<>();
    private final List<PathFragment> quoteIncludes = new ArrayList<>();
    private final List<PathFragment> strictDependencyIncludes = new ArrayList<>();
    private final List<CcCompilationContext> directCcCompilationContexts = new ArrayList<>();
    private final List<CcCompilationContext> ccCompilationContexts = new ArrayList<>();
    private final List<CcCompilationContext> implementationCcCompilationContexts =
        new ArrayList<>();

    Builder() {}

    @CanIgnoreReturnValue
    public Builder addDefines(Iterable<String> defines) {
      Iterables.addAll(this.defines, defines);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addPublicHeaders(Iterable<Artifact> headers) {
      Iterables.addAll(this.publicHeaders, headers);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addPublicTextualHeaders(Iterable<Artifact> headers) {
      Iterables.addAll(this.publicTextualHeaders, headers);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addPrivateHeaders(Iterable<Artifact> headers) {
      Iterables.addAll(this.privateHeaders, headers);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addIncludes(Iterable<PathFragment> includes) {
      Iterables.addAll(this.includes, includes);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addSystemIncludes(Iterable<PathFragment> includes) {
      Iterables.addAll(this.systemIncludes, includes);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addQuoteIncludes(Iterable<PathFragment> includes) {
      Iterables.addAll(this.quoteIncludes, includes);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addObjcProviders(Iterable<StarlarkInfoWithSchema> objcProviders)
        throws EvalException {
      for (StarlarkInfoWithSchema objcProvider : objcProviders) {
        this.strictDependencyIncludes.addAll(
            Depset.cast(objcProvider.getValue("strict_include"), String.class, "strict_include")
                .toList()
                .stream()
                .map(PathFragment::create)
                .collect(ImmutableList.toImmutableList()));
      }
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addCcCompilationContexts(Iterable<CcCompilationContext> ccCompilationContexts) {
      Iterables.addAll(this.ccCompilationContexts, ccCompilationContexts);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addDirectCcCompilationContexts(
        Iterable<CcCompilationContext> ccCompilationContexts) {
      Iterables.addAll(this.directCcCompilationContexts, ccCompilationContexts);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addImplementationCcCompilationContexts(
        Iterable<CcCompilationContext> ccCompilationContexts) {
      Iterables.addAll(this.implementationCcCompilationContexts, ccCompilationContexts);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addCcCompilationContext(CcCompilationContext ccCompilationContext) {
      this.ccCompilationContexts.add(ccCompilationContext);
      return this;
    }

    ObjcCompilationContext build() {
      return new ObjcCompilationContext(
          defines,
          publicHeaders,
          publicTextualHeaders,
          privateHeaders,
          includes,
          systemIncludes,
          quoteIncludes,
          strictDependencyIncludes,
          directCcCompilationContexts,
          ccCompilationContexts,
          implementationCcCompilationContexts);
    }
  }
}
