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

package com.google.devtools.build.lib.rules.objc;


import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.cpp.CcCompilationContext;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.ArrayList;
import java.util.List;
import net.starlark.java.eval.StarlarkValue;

/**
 * Contains information common to multiple objc_* rules, and provides a unified API for extracting
 * and accessing it.
 */
// TODO(bazel-team): Decompose and subsume area-specific logic and data into the various *Support
// classes. Make sure to distinguish rule output (providers, runfiles, ...) from intermediate,
// rule-internal information. Any provider created by a rule should not be read, only published.
public final class ObjcCommon implements StarlarkValue {

  /** Filters fileset artifacts out of a group of artifacts. */
  private static ImmutableList<Artifact> filterFileset(Iterable<Artifact> artifacts) {
    ImmutableList.Builder<Artifact> inputs = ImmutableList.<Artifact>builder();
    for (Artifact artifact : artifacts) {
      if (!artifact.isFileset()) {
        inputs.add(artifact);
      }
    }
    return inputs.build();
  }

  static class Builder {
    private final RuleContext context;
    private final BuildConfigurationValue buildConfiguration;
    private Optional<CompilationAttributes> compilationAttributes = Optional.absent();
    private Iterable<ObjcProvider> objcProviders = ImmutableList.of();
    private final List<CcCompilationContext> ccCompilationContexts = new ArrayList<>();
    private final List<CcLinkingContext> ccLinkingContexts = new ArrayList<>();

    /**
     * Builder for {@link ObjcCommon} obtaining attribute data from the rule context and
     * configuration data from the given configuration object for use in situations where a single
     * target's outputs are under multiple configurations.
     */
    Builder(RuleContext context, BuildConfigurationValue buildConfiguration)
        throws InterruptedException {
      this.context = Preconditions.checkNotNull(context);
      this.buildConfiguration = Preconditions.checkNotNull(buildConfiguration);
    }

    @CanIgnoreReturnValue
    public Builder setCompilationAttributes(CompilationAttributes baseCompilationAttributes) {
      Preconditions.checkState(
          !this.compilationAttributes.isPresent(),
          "compilationAttributes is already set to: %s",
          this.compilationAttributes);
      this.compilationAttributes = Optional.of(baseCompilationAttributes);
      return this;
    }

    @CanIgnoreReturnValue
    Builder addCcCompilationContexts(Iterable<CcInfo> ccInfos) {
      ccInfos.forEach(ccInfo -> ccCompilationContexts.add(ccInfo.getCcCompilationContext()));
      return this;
    }

    @CanIgnoreReturnValue
    Builder addCcLinkingContexts(Iterable<CcInfo> ccInfos) {
      ccInfos.forEach(ccInfo -> ccLinkingContexts.add(ccInfo.getCcLinkingContext()));
      return this;
    }

    @CanIgnoreReturnValue
    Builder addCcInfos(Iterable<CcInfo> ccInfos) {
      addCcCompilationContexts(ccInfos);
      addCcLinkingContexts(ccInfos);
      return this;
    }

    @CanIgnoreReturnValue
    Builder addDeps(List<? extends TransitiveInfoCollection> deps) {
      ImmutableList.Builder<ObjcProvider> objcProviders = ImmutableList.builder();
      ImmutableList.Builder<CcInfo> ccInfos = ImmutableList.builder();

      for (TransitiveInfoCollection dep : deps) {
        ObjcProvider objcProvider = dep.get(ObjcProvider.STARLARK_CONSTRUCTOR);
        if (objcProvider != null) {
          objcProviders.add(objcProvider);
        }
        CcInfo ccInfo = dep.get(CcInfo.PROVIDER);
        if (ccInfo != null) {
          ccInfos.add(ccInfo);
        }
      }

      addObjcProviders(objcProviders.build());
      addCcInfos(ccInfos.build());

      return this;
    }

    /**
     * Add providers which will be exposed both to the declaring rule and to any dependers on the
     * declaring rule.
     */
    @CanIgnoreReturnValue
    Builder addObjcProviders(Iterable<ObjcProvider> objcProviders) {
      this.objcProviders = Iterables.concat(this.objcProviders, objcProviders);
      return this;
    }

    ObjcCommon build() {
      ImmutableList<CcCompilationContext> ccCompilationContexts =
          ImmutableList.copyOf(this.ccCompilationContexts);
      ImmutableList<CcLinkingContext> ccLinkingContexts =
          ImmutableList.copyOf(this.ccLinkingContexts);

      ObjcCompilationContext.Builder objcCompilationContextBuilder =
          ObjcCompilationContext.builder();

      ObjcProvider.Builder objcProvider = new ObjcProvider.Builder();

      objcProvider
          .addTransitiveAndPropagate(objcProviders);

      objcCompilationContextBuilder
          .addObjcProviders(objcProviders)
          // TODO(bazel-team): This pulls in stl via
          // CcCompilationHelper.getStlCcCompilationContext(), but probably shouldn't.
          .addCcCompilationContexts(ccCompilationContexts);

      if (compilationAttributes.isPresent()) {
        CompilationAttributes attributes = compilationAttributes.get();
        PathFragment usrIncludeDir = PathFragment.create(AppleToolchain.sdkDir() + "/usr/include/");
        Iterable<PathFragment> sdkIncludes =
            Iterables.transform(
                attributes.sdkIncludes().toList(), (p) -> usrIncludeDir.getRelative(p));
        objcCompilationContextBuilder
            .addPublicHeaders(filterFileset(attributes.hdrs().toList()))
            .addPublicTextualHeaders(filterFileset(attributes.textualHdrs().toList()))
            .addDefines(attributes.defines())
            .addIncludes(
                attributes
                    .headerSearchPaths(
                        buildConfiguration.getGenfilesFragment(context.getRepository()))
                    .toList())
            .addIncludes(sdkIncludes);
      }

      ObjcCompilationContext objcCompilationContext = objcCompilationContextBuilder.build();

      return new ObjcCommon(objcProvider.build(), objcCompilationContext, ccLinkingContexts);
    }
  }

  private final ObjcProvider objcProvider;
  private final ObjcCompilationContext objcCompilationContext;
  private final ImmutableList<CcLinkingContext> ccLinkingContexts;

  private ObjcCommon(
      ObjcProvider objcProvider,
      ObjcCompilationContext objcCompilationContext,
      ImmutableList<CcLinkingContext> ccLinkingContexts) {
    this.objcProvider = Preconditions.checkNotNull(objcProvider);
    this.objcCompilationContext = Preconditions.checkNotNull(objcCompilationContext);
    this.ccLinkingContexts = Preconditions.checkNotNull(ccLinkingContexts);
  }

  public ObjcProvider getObjcProvider() {
    return objcProvider;
  }

  public ObjcCompilationContext getObjcCompilationContext() {
    return objcCompilationContext;
  }

  public ImmutableList<CcLinkingContext> getCcLinkingContexts() {
    return ccLinkingContexts;
  }

  public CcCompilationContext createCcCompilationContext() {
    return objcCompilationContext.createCcCompilationContext();
  }

  public CcLinkingContext createCcLinkingContext() {
    return CcLinkingContext.merge(ccLinkingContexts);
  }

  public CcInfo createCcInfo() {
    return CcInfo.builder()
        .setCcCompilationContext(createCcCompilationContext())
        .setCcLinkingContext(createCcLinkingContext())
        .build();
  }
}
