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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.Collection;
import java.util.List;
import java.util.Objects;

/**
 * Parameters to be passed to the linker.
 *
 * <p>The parameters concerned are the link options (strings) passed to the linker, linkstamps, a
 * list of libraries to be linked in, and a list of libraries to build at link time.
 *
 * <p>Items in the collections are stored in nested sets. Link options and libraries are stored in
 * link order (preorder) and linkstamps are sorted.
 */
public final class CcLinkParams {
  private final NestedSet<ImmutableList<String>> linkOpts;
  private final NestedSet<Linkstamp> linkstamps;
  private final NestedSet<LibraryToLink> libraries;
  private final ExtraLinkTimeLibraries extraLinkTimeLibraries;

  private CcLinkParams(NestedSet<ImmutableList<String>> linkOpts,
                       NestedSet<Linkstamp> linkstamps,
                       NestedSet<LibraryToLink> libraries,
                       ExtraLinkTimeLibraries extraLinkTimeLibraries) {
    this.linkOpts = linkOpts;
    this.linkstamps = linkstamps;
    this.libraries = libraries;
    this.extraLinkTimeLibraries = extraLinkTimeLibraries;
  }

  /**
   * @return the linkopts
   */
  public NestedSet<ImmutableList<String>> getLinkopts() {
    return linkOpts;
  }

  public ImmutableList<String> flattenedLinkopts() {
    return ImmutableList.copyOf(Iterables.concat(linkOpts));
  }

  /**
   * @return the linkstamps
   */
  public NestedSet<Linkstamp> getLinkstamps() {
    return linkstamps;
  }

  /**
   * @return the libraries
   */
  public NestedSet<LibraryToLink> getLibraries() {
    return libraries;
  }

  /**
   * The extra link time libraries; will be null if there are no such libraries.
   */
  public ExtraLinkTimeLibraries getExtraLinkTimeLibraries() {
    return extraLinkTimeLibraries;
  }

  public static final Builder builder(boolean linkingStatically, boolean linkShared) {
    return new Builder(linkingStatically, linkShared);
  }

  /**
   * Builder for {@link CcLinkParams}.
   *
 *
   */
  public static final class Builder {

    /**
     * linkingStatically is true when we're linking this target in either FULLY STATIC mode
     * (linkopts=["-static"]) or MOSTLY STATIC mode (linkstatic=1). When this is true, we want to
     * use static versions of any libraries that this target depends on (except possibly system
     * libraries, which are not handled by CcLinkParams). When this is false, we want to use dynamic
     * versions of any libraries that this target depends on.
     */
    private final boolean linkingStatically;

    /**
     * linkShared is true when we're linking with "-shared" (linkshared=1).
     */
    private final boolean linkShared;

    private ImmutableList.Builder<String> localLinkoptsBuilder = ImmutableList.builder();

    private final NestedSetBuilder<ImmutableList<String>> linkOptsBuilder =
        NestedSetBuilder.linkOrder();
    private final NestedSetBuilder<Linkstamp> linkstampsBuilder =
        NestedSetBuilder.compileOrder();
    private final NestedSetBuilder<LibraryToLink> librariesBuilder =
        NestedSetBuilder.linkOrder();

    /**
     * A builder for the list of link time libraries.  Most builds
     * won't have any such libraries, so save space by leaving the
     * default as null.
     */
    private ExtraLinkTimeLibraries.Builder extraLinkTimeLibrariesBuilder = null;

    private boolean built = false;

    private Builder(boolean linkingStatically, boolean linkShared) {
      this.linkingStatically = linkingStatically;
      this.linkShared = linkShared;
    }

    /**
     * Build a {@link CcLinkParams} object.
     */
    public CcLinkParams build() {
      Preconditions.checkState(!built);
      // Not thread-safe, but builders should not be shared across threads.
      built = true;
      ImmutableList<String> localLinkopts = localLinkoptsBuilder.build();
      if (!localLinkopts.isEmpty()) {
        linkOptsBuilder.add(localLinkopts);
      }
      ExtraLinkTimeLibraries extraLinkTimeLibraries = null;
      if (extraLinkTimeLibrariesBuilder != null) {
        extraLinkTimeLibraries = extraLinkTimeLibrariesBuilder.build();
      }
      return new CcLinkParams(linkOptsBuilder.build(), linkstampsBuilder.build(),
          librariesBuilder.build(), extraLinkTimeLibraries);
    }

    public boolean add(CcLinkParamsStore store) {
      if (store != null) {
        CcLinkParams args = store.get(linkingStatically, linkShared);
        addTransitiveArgs(args);
      }
      return store != null;
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
     * Includes link parameters from a dependency target.
     *
     * <p>The target should implement {@link CcLinkParamsProvider}. If it does not,
     * the method does not do anything.
     */
    public Builder addTransitiveTarget(TransitiveInfoCollection target) {
      return addTransitiveProvider(target.getProvider(CcLinkParamsProvider.class));
    }

    /**
     * Includes link parameters from a dependency target. The target is checked for the given
     * mappings in the order specified, and the first mapping that returns a non-null result is
     * added.
     */
    @SafeVarargs
    public final Builder addTransitiveTarget(TransitiveInfoCollection target,
        Function<TransitiveInfoCollection, CcLinkParamsStore> firstMapping,
        @SuppressWarnings("unchecked") // Java arrays don't preserve generic arguments.
        Function<TransitiveInfoCollection, CcLinkParamsStore>... remainingMappings) {
      if (add(firstMapping.apply(target))) {
        return this;
      }
      for (Function<TransitiveInfoCollection, CcLinkParamsStore> mapping : remainingMappings) {
        if (add(mapping.apply(target))) {
          return this;
        }
      }
      return this;
    }

    /**
     * Includes link parameters from a CcLinkParamsProvider provider.
     */
    public Builder addTransitiveProvider(CcLinkParamsProvider provider) {
      if (provider != null) {
        add(provider.getCcLinkParamsStore());
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
        Function<TransitiveInfoCollection, CcLinkParamsStore> firstMapping,
        @SuppressWarnings("unchecked")  // Java arrays don't preserve generic arguments.
        Function<TransitiveInfoCollection, CcLinkParamsStore>... remainingMappings) {
      for (TransitiveInfoCollection target : targets) {
        addTransitiveTarget(target, firstMapping, remainingMappings);
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
      if (args.getExtraLinkTimeLibraries() != null) {
        if (extraLinkTimeLibrariesBuilder == null) {
          extraLinkTimeLibrariesBuilder = ExtraLinkTimeLibraries.builder();
        }
        extraLinkTimeLibrariesBuilder.addTransitive(args.getExtraLinkTimeLibraries());
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

    /**
     * Adds a collection of linkstamps.
     */
    public Builder addLinkstamps(Iterable<Artifact> linkstamps, CppCompilationContext context) {
      ImmutableList<Artifact> declaredIncludeSrcs =
          ImmutableList.copyOf(context.getDeclaredIncludeSrcs());
      for (Artifact linkstamp : linkstamps) {
        linkstampsBuilder.add(new Linkstamp(linkstamp, declaredIncludeSrcs));
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
     * Processes typical dependencies a C/C++ library.
     *
     * <p>A helper method that processes getValues() and merges contents of
     * getPreferredLibraries() and getLinkOpts() into the current link params
     * object.
     */
    public Builder addCcLibrary(RuleContext context, boolean neverlink, List<String> linkopts,
        CcLinkingOutputs linkingOutputs) {
      addTransitiveTargets(
          context.getPrerequisites("deps", Mode.TARGET),
          CcLinkParamsProvider.TO_LINK_PARAMS, CcSpecificLinkParamsProvider.TO_LINK_PARAMS);

      if (!neverlink) {
        addLibraries(linkingOutputs.getPreferredLibraries(linkingStatically,
            linkShared || context.getFragment(CppConfiguration.class).forcePic()));
        addLinkOpts(linkopts);
      }
      return this;
    }
  }

  /**
   * A linkstamp that also knows about its declared includes.
   *
   * <p>This object is required because linkstamp files may include other headers which
   * will have to be provided during compilation.
   */
  public static final class Linkstamp {
    private final Artifact artifact;
    private final ImmutableList<Artifact> declaredIncludeSrcs;

    private Linkstamp(Artifact artifact, ImmutableList<Artifact> declaredIncludeSrcs) {
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
    public ImmutableList<Artifact> getDeclaredIncludeSrcs() {
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

  /**
   * Empty CcLinkParams.
   */
  public static final CcLinkParams EMPTY = new CcLinkParams(
      NestedSetBuilder.<ImmutableList<String>>emptySet(Order.LINK_ORDER),
      NestedSetBuilder.<Linkstamp>emptySet(Order.COMPILE_ORDER),
      NestedSetBuilder.<LibraryToLink>emptySet(Order.LINK_ORDER),
      null);
}
