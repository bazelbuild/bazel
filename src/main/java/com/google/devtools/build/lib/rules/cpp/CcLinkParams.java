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

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcLinkParamsApi;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import java.util.Collection;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * Parameters to be passed to the linker.
 *
 * <p>The parameters concerned are the link options (strings) passed to the linker, linkstamps, a
 * list of libraries to be linked in, and a list of libraries to build at link time.
 *
 * <p>Items in the collections are stored in nested sets. Link options and libraries are stored in
 * link order (preorder) and linkstamps are sorted.
 */
@AutoCodec
public final class CcLinkParams implements CcLinkParamsApi {
  /**
   * A list of link options contributed by a single configured target.
   *
   * <p><b>WARNING:</b> Do not implement {@code #equals()} in the obvious way. This class must be
   * checked for equality by object identity because otherwise if two configured targets contribute
   * the same link options, they will be de-duplicated, which is not the desirable behavior.
   */
  @AutoCodec
  @Immutable
  public static final class LinkOptions {
    private final ImmutableList<String> linkOptions;

    @VisibleForSerialization
    LinkOptions(Iterable<String> linkOptions) {
      this.linkOptions = ImmutableList.copyOf(linkOptions);
    }

    public ImmutableList<String> get() {
      return linkOptions;
    }

    public static LinkOptions of(Iterable<String> linkOptions) {
      return new LinkOptions(linkOptions);
    }
  }

  private final NestedSet<LinkOptions> linkOpts;
  private final NestedSet<Linkstamp> linkstamps;
  private final NestedSet<LibraryToLink> libraries;
  private final NestedSet<Artifact> dynamicLibrariesForRuntime;
  private final ExtraLinkTimeLibraries extraLinkTimeLibraries;
  private final NestedSet<Artifact> nonCodeInputs;

  @AutoCodec.Instantiator
  @VisibleForSerialization
  CcLinkParams(
      NestedSet<LinkOptions> linkOpts,
      NestedSet<Linkstamp> linkstamps,
      NestedSet<LibraryToLink> libraries,
      NestedSet<Artifact> dynamicLibrariesForRuntime,
      ExtraLinkTimeLibraries extraLinkTimeLibraries,
      NestedSet<Artifact> nonCodeInputs) {
    this.linkOpts = linkOpts;
    this.linkstamps = linkstamps;
    this.libraries = libraries;
    this.dynamicLibrariesForRuntime = dynamicLibrariesForRuntime;
    this.extraLinkTimeLibraries = extraLinkTimeLibraries;
    this.nonCodeInputs = nonCodeInputs;
  }

  /**
   * Returns the linkopts
   */
  public NestedSet<LinkOptions> getLinkopts() {
    return linkOpts;
  }

  public ImmutableList<String> flattenedLinkopts() {
    return ImmutableList.copyOf(Iterables.concat(Iterables.transform(linkOpts, LinkOptions::get)));
  }

  @Override
  public SkylarkNestedSet getSkylarkLinkopts() {
    // TODO(plf): Shouldn't flatten nested set. Remove LinkOptions class and just have a nested set
    // of strings.
    return SkylarkNestedSet.of(
        String.class, NestedSetBuilder.wrap(Order.COMPILE_ORDER, flattenedLinkopts()));
  }

  /**
   * Returns the linkstamps
   */
  public NestedSet<Linkstamp> getLinkstamps() {
    return linkstamps;
  }

  /**
   * Returns the libraries
   */
  public NestedSet<LibraryToLink> getLibraries() {
    return libraries;
  }

  @Override
  public SkylarkNestedSet getSkylarkLibrariesToLink() {
    return SkylarkNestedSet.of(LibraryToLink.class, libraries);
  }

  /** Returns the dynamicLibrariesForRuntime. */
  public NestedSet<Artifact> getDynamicLibrariesForRuntime() {
    return dynamicLibrariesForRuntime;
  }

  @Override
  public SkylarkNestedSet getSkylarkDynamicLibrariesForRuntime() {
    return SkylarkNestedSet.of(Artifact.class, dynamicLibrariesForRuntime);
  }

  /**
   * The extra link time libraries; will be null if there are no such libraries.
   */
  public @Nullable ExtraLinkTimeLibraries getExtraLinkTimeLibraries() {
    return extraLinkTimeLibraries;
  }

  /**
   * Returns the non-code inputs, e.g. linker scripts; will be null if none.
   */
  public @Nullable NestedSet<Artifact> getNonCodeInputs() {
    return nonCodeInputs;
  }

  public static final Builder builder(boolean linkingStatically, boolean linkShared) {
    return new Builder(linkingStatically, linkShared);
  }

  public static final Builder builder() {
    return new Builder();
  }

  /**
   * Builder for {@link CcLinkParams}.
   */
  public static final class Builder {

    /**
     * linkingStatically is true when we're linking this target in either FULLY STATIC mode
     * (linkopts=["-static"]) or MOSTLY STATIC mode (linkstatic=1). When this is true, we want to
     * use static versions of any libraries that this target depends on (except possibly system
     * libraries, which are not handled by CcLinkParams). When this is false, we want to use dynamic
     * versions of any libraries that this target depends on.
     */
    private boolean linkingStatically;

    /** linkShared is true when we're linking with "-shared" (linkshared=1). */
    private boolean linkShared;

    // TODO(plf): Ideally the two booleans above are removed from this Builder. We would pass the
    // specific instances of CcLinkParams that are needed from transitive dependencies instead of
    // calling the convenience methods that dig them out from the CcLinkingInfo using these
    // booleans.
    private boolean linkingStaticallyLinkSharedSet;

    private ImmutableList.Builder<String> localLinkoptsBuilder = ImmutableList.builder();

    private final NestedSetBuilder<LinkOptions> linkOptsBuilder =
        NestedSetBuilder.linkOrder();
    private final NestedSetBuilder<Linkstamp> linkstampsBuilder =
        NestedSetBuilder.compileOrder();
    private final NestedSetBuilder<LibraryToLink> librariesBuilder =
        NestedSetBuilder.linkOrder();
    private final NestedSetBuilder<Artifact> dynamicLibrariesForRuntimeBuilder =
        NestedSetBuilder.stableOrder();

    /**
     * A builder for the list of link time libraries.  Most builds
     * won't have any such libraries, so save space by leaving the
     * default as null.
     */
    private ExtraLinkTimeLibraries.Builder extraLinkTimeLibrariesBuilder = null;

    private NestedSetBuilder<Artifact> nonCodeInputsBuilder = null;

    private boolean built = false;

    /** The static builder methods of {@link CcLinkParams} should be used for instantiation. */
    private Builder(boolean linkingStatically, boolean linkShared) {
      this.linkingStatically = linkingStatically;
      this.linkShared = linkShared;
      this.linkingStaticallyLinkSharedSet = true;
    }

    private Builder() {}

    /**
     * Builds a {@link CcLinkParams} object.
     */
    public CcLinkParams build() {
      Preconditions.checkState(!built);
      // Not thread-safe, but builders should not be shared across threads.
      built = true;
      ImmutableList<String> localLinkopts = localLinkoptsBuilder.build();
      if (!localLinkopts.isEmpty()) {
        linkOptsBuilder.add(LinkOptions.of(localLinkopts));
      }
      ExtraLinkTimeLibraries extraLinkTimeLibraries = null;
      if (extraLinkTimeLibrariesBuilder != null) {
        extraLinkTimeLibraries = extraLinkTimeLibrariesBuilder.build();
      }
      NestedSet<Artifact> nonCodeInputs = null;
      if (nonCodeInputsBuilder != null) {
        nonCodeInputs = nonCodeInputsBuilder.build();
      }
      return new CcLinkParams(
          linkOptsBuilder.build(),
          linkstampsBuilder.build(),
          librariesBuilder.build(),
          dynamicLibrariesForRuntimeBuilder.build(),
          extraLinkTimeLibraries,
          nonCodeInputs);
    }

    public boolean add(CcLinkingInfo ccLinkingInfo) {
      Preconditions.checkState(linkingStaticallyLinkSharedSet);
      if (ccLinkingInfo != null) {
        CcLinkParams args = ccLinkingInfo.getCcLinkParams(linkingStatically, linkShared);
        addTransitiveArgs(args);
      }
      return ccLinkingInfo != null;
    }

    /**
     * Includes link parameters from a collection of dependency targets.
     */
    public Builder addTransitiveTargets(Iterable<? extends TransitiveInfoCollection> targets) {
      for (TransitiveInfoCollection target : targets) {
        addTransitiveTarget(target);
      }
      return this;
    }

    /**
     * Includes link parameters from the given targets. Each target is checked for the given
     * mappings in the order specified, and the first mapping that returns a non-null result is
     * added.
     */
    @SafeVarargs
    public final Builder addTransitiveTargets(
        Iterable<? extends TransitiveInfoCollection> targets,
        Function<TransitiveInfoCollection, CcLinkingInfo> firstMapping,
        @SuppressWarnings("unchecked") // Java arrays don't preserve generic arguments.
            Function<TransitiveInfoCollection, CcLinkingInfo>... remainingMappings) {
      for (TransitiveInfoCollection target : targets) {
        addTransitiveTarget(target, firstMapping, remainingMappings);
      }
      return this;
    }

    /**
     * Includes link parameters from a dependency target.
     *
     * <p>The target should implement {@link CcLinkingInfo}. If it does not, the method does not do
     * anything.
     */
    public Builder addTransitiveTarget(TransitiveInfoCollection target) {
      CcInfo ccInfo = target.get(CcInfo.PROVIDER);
      if (ccInfo != null) {
        add(ccInfo.getCcLinkingInfo());
      }
      return this;
    }

    /**
     * Includes link parameters from a dependency target. The target is checked for the given
     * mappings in the order specified, and the first mapping that returns a non-null result is
     * added.
     */
    @SafeVarargs
    public final Builder addTransitiveTarget(
        TransitiveInfoCollection target,
        Function<TransitiveInfoCollection, CcLinkingInfo> firstMapping,
        @SuppressWarnings("unchecked") // Java arrays don't preserve generic arguments.
            Function<TransitiveInfoCollection, CcLinkingInfo>... remainingMappings) {
      if (add(firstMapping.apply(target))) {
        return this;
      }
      for (Function<TransitiveInfoCollection, CcLinkingInfo> mapping : remainingMappings) {
        if (add(mapping.apply(target))) {
          return this;
        }
      }
      return this;
    }

    /**
     * Merges the other {@link CcLinkParams} object into this one.
     */
    public Builder addTransitiveArgs(CcLinkParams args) {
      linkOptsBuilder.addTransitive(args.getLinkopts());
      linkstampsBuilder.addTransitive(args.getLinkstamps());
      librariesBuilder.addTransitive(args.getLibraries());
      dynamicLibrariesForRuntimeBuilder.addTransitive(args.getDynamicLibrariesForRuntime());
      if (args.getExtraLinkTimeLibraries() != null) {
        if (extraLinkTimeLibrariesBuilder == null) {
          extraLinkTimeLibrariesBuilder = ExtraLinkTimeLibraries.builder();
        }
        extraLinkTimeLibrariesBuilder.addTransitive(args.getExtraLinkTimeLibraries());
      }
      if (args.getNonCodeInputs() != null) {
        if (nonCodeInputsBuilder == null) {
          nonCodeInputsBuilder = NestedSetBuilder.linkOrder();
        }
        nonCodeInputsBuilder.addTransitive(args.getNonCodeInputs());
      }
      return this;
    }

    /**
     * Adds a collection of link options.
     */
    public Builder addLinkOpts(Collection<String> linkOpts) {
      localLinkoptsBuilder.addAll(linkOpts);
      return this;
    }

    /** Adds a collection of linkstamps. */
    public Builder addLinkstamps(
        NestedSet<Artifact> linkstamps, CcCompilationContext ccCompilationContext) {
      for (Artifact linkstamp : linkstamps) {
        linkstampsBuilder.add(
            new Linkstamp(linkstamp, ccCompilationContext.getDeclaredIncludeSrcs()));
      }
      return this;
    }

    /**
     * Adds a library artifact.
     */
    public Builder addLibrary(LibraryToLink library) {
      librariesBuilder.add(library);
      return this;
    }

    /**
     * Adds a collection of library artifacts.
     */
    public Builder addLibraries(Iterable<LibraryToLink> libraries) {
      librariesBuilder.addAll(libraries);
      return this;
    }

    /** Adds a collection of library artifacts. */
    public Builder addDynamicLibrariesForRuntime(Iterable<Artifact> libraries) {
      dynamicLibrariesForRuntimeBuilder.addAll(libraries);
      return this;
    }

    /**
     * Adds an extra link time library, a library that is actually
     * built at link time.
     */
    public Builder addExtraLinkTimeLibrary(ExtraLinkTimeLibrary e) {
      if (extraLinkTimeLibrariesBuilder == null) {
        extraLinkTimeLibrariesBuilder = ExtraLinkTimeLibraries.builder();
      }
      extraLinkTimeLibrariesBuilder.add(e);
      return this;
    }

    /**
     * Adds a collection of non-code inputs.
     */
    public Builder addNonCodeInputs(Iterable<Artifact> nonCodeInputs) {
      if (nonCodeInputsBuilder == null) {
        nonCodeInputsBuilder = NestedSetBuilder.linkOrder();
      }
      nonCodeInputsBuilder.addAll(nonCodeInputs);
      return this;
    }

    /** Processes typical dependencies of a C/C++ library. */
    public Builder addCcLibrary(RuleContext context) {
      addTransitiveTargets(
          context.getPrerequisites("deps", Mode.TARGET),
          x -> {
            if (x.get(CcInfo.PROVIDER) == null) {
              return null;
            }
            return x.get(CcInfo.PROVIDER).getCcLinkingInfo();
          });
      return this;
    }
  }

  /**
   * A linkstamp that also knows about its declared includes.
   *
   * <p>This object is required because linkstamp files may include other headers which will have to
   * be provided during compilation.
   */
  @AutoCodec
  public static final class Linkstamp {
    private final Artifact artifact;
    private final NestedSet<Artifact> declaredIncludeSrcs;

    @VisibleForSerialization
    Linkstamp(Artifact artifact, NestedSet<Artifact> declaredIncludeSrcs) {
      this.artifact = Preconditions.checkNotNull(artifact);
      this.declaredIncludeSrcs = Preconditions.checkNotNull(declaredIncludeSrcs);
    }

    /**
     * Returns the linkstamp artifact.
     */
    public Artifact getArtifact() {
      return artifact;
    }

    /**
     * Returns the declared includes.
     */
    public NestedSet<Artifact> getDeclaredIncludeSrcs() {
      return declaredIncludeSrcs;
    }

    @Override
    public int hashCode() {
      return Objects.hash(artifact, declaredIncludeSrcs);
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof Linkstamp)) {
        return false;
      }
      Linkstamp other = (Linkstamp) obj;
      return artifact.equals(other.artifact)
          && declaredIncludeSrcs.equals(other.declaredIncludeSrcs);
    }
  }

  /** Empty CcLinkParams. */
  public static final CcLinkParams EMPTY =
      new CcLinkParams(
          NestedSetBuilder.<LinkOptions>emptySet(Order.LINK_ORDER),
          NestedSetBuilder.<Linkstamp>emptySet(Order.COMPILE_ORDER),
          NestedSetBuilder.<LibraryToLink>emptySet(Order.LINK_ORDER),
          NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
          null,
          null);
}
