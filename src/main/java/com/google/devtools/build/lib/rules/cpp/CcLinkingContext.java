// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Joiner;
import com.google.common.base.MoreObjects;
import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.SymbolGenerator;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcLinkingContextApi;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.util.Fingerprint;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/** Structure of CcLinkingContext. */
public class CcLinkingContext implements CcLinkingContextApi<Artifact> {
  public static final CcLinkingContext EMPTY = builder().build();

  /** A list of link options contributed by a single configured target/aspect. */
  @Immutable
  public static final class LinkOptions {
    private final ImmutableList<String> linkOptions;
    private final Object symbolForEquality;

    private LinkOptions(Iterable<String> linkOptions, Object symbolForEquality) {
      this.linkOptions = ImmutableList.copyOf(linkOptions);
      this.symbolForEquality = Preconditions.checkNotNull(symbolForEquality);
    }

    public ImmutableList<String> get() {
      return linkOptions;
    }

    public static LinkOptions of(Iterable<String> linkOptions, SymbolGenerator<?> symbolGenerator) {
      return new LinkOptions(linkOptions, symbolGenerator.generate());
    }

    @Override
    public int hashCode() {
      // Symbol is sufficient for equality check.
      return symbolForEquality.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof LinkOptions)) {
        return false;
      }
      LinkOptions that = (LinkOptions) obj;
      if (!this.symbolForEquality.equals(that.symbolForEquality)) {
        return false;
      }
      if (this.linkOptions.equals(that.linkOptions)) {
        return true;
      }
      BugReport.sendBugReport(
          new IllegalStateException(
              "Unexpected inequality with equal symbols: " + this + ", " + that));
      return false;
    }

    @Override
    public String toString() {
      return '[' + Joiner.on(",").join(linkOptions) + "] (owner: " + symbolForEquality;
    }
  }

  /**
   * A linkstamp that also knows about its declared includes.
   *
   * <p>This object is required because linkstamp files may include other headers which will have to
   * be provided during compilation.
   */
  public static final class Linkstamp {
    private final Artifact artifact;
    private final NestedSet<Artifact> declaredIncludeSrcs;
    private final byte[] nestedDigest;

    // TODO(janakr): if action key context is not available, the digest can be computed lazily,
    // only if we are doing an equality comparison and artifacts are equal. That should never
    // happen, so doing an expensive digest should be ok then. If this is ever moved to Starlark
    // and Starlark doesn't support custom equality or amortized deep equality of nested sets, a
    // Symbol can be used as an equality proxy, similar to what LinkOptions does above.
    Linkstamp(
        Artifact artifact,
        NestedSet<Artifact> declaredIncludeSrcs,
        ActionKeyContext actionKeyContext) {
      this.artifact = Preconditions.checkNotNull(artifact);
      this.declaredIncludeSrcs = Preconditions.checkNotNull(declaredIncludeSrcs);
      Fingerprint fp = new Fingerprint();
      actionKeyContext.addNestedSetToFingerprint(fp, this.declaredIncludeSrcs);
      nestedDigest = fp.digestAndReset();
    }

    /** Returns the linkstamp artifact. */
    public Artifact getArtifact() {
      return artifact;
    }

    /** Returns the declared includes. */
    public NestedSet<Artifact> getDeclaredIncludeSrcs() {
      return declaredIncludeSrcs;
    }

    @Override
    public int hashCode() {
      // Artifact should be enough to disambiguate basically all the time.
      return artifact.hashCode();
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
          && Arrays.equals(this.nestedDigest, other.nestedDigest);
    }
  }

  private final NestedSet<LibraryToLink> libraries;
  private final NestedSet<LinkOptions> userLinkFlags;
  private final NestedSet<Linkstamp> linkstamps;
  private final NestedSet<Artifact> nonCodeInputs;
  private final ExtraLinkTimeLibraries extraLinkTimeLibraries;

  public CcLinkingContext(
      NestedSet<LibraryToLink> libraries,
      NestedSet<LinkOptions> userLinkFlags,
      NestedSet<Linkstamp> linkstamps,
      NestedSet<Artifact> nonCodeInputs,
      ExtraLinkTimeLibraries extraLinkTimeLibraries) {
    this.libraries = libraries;
    this.userLinkFlags = userLinkFlags;
    this.linkstamps = linkstamps;
    this.nonCodeInputs = nonCodeInputs;
    this.extraLinkTimeLibraries = extraLinkTimeLibraries;
  }

  public static CcLinkingContext merge(List<CcLinkingContext> ccLinkingContexts) {
    Builder mergedCcLinkingContext = CcLinkingContext.builder();
    ExtraLinkTimeLibraries.Builder mergedExtraLinkTimeLibraries = ExtraLinkTimeLibraries.builder();
    for (CcLinkingContext ccLinkingContext : ccLinkingContexts) {
      mergedCcLinkingContext
          .addLibraries(ccLinkingContext.getLibraries())
          .addUserLinkFlags(ccLinkingContext.getUserLinkFlags())
          .addLinkstamps(ccLinkingContext.getLinkstamps())
          .addNonCodeInputs(ccLinkingContext.getNonCodeInputs());
      if (ccLinkingContext.getExtraLinkTimeLibraries() != null) {
        mergedExtraLinkTimeLibraries.addTransitive(ccLinkingContext.getExtraLinkTimeLibraries());
      }
    }
    mergedCcLinkingContext.setExtraLinkTimeLibraries(mergedExtraLinkTimeLibraries.build());
    return mergedCcLinkingContext.build();
  }

  public List<Artifact> getStaticModeParamsForExecutableLibraries() {
    ImmutableList.Builder<Artifact> libraryListBuilder = ImmutableList.builder();
    for (LibraryToLink libraryToLink : getLibraries()) {
      if (libraryToLink.getStaticLibrary() != null) {
        libraryListBuilder.add(libraryToLink.getStaticLibrary());
      } else if (libraryToLink.getPicStaticLibrary() != null) {
        libraryListBuilder.add(libraryToLink.getPicStaticLibrary());
      } else if (libraryToLink.getInterfaceLibrary() != null) {
        libraryListBuilder.add(libraryToLink.getInterfaceLibrary());
      } else {
        libraryListBuilder.add(libraryToLink.getDynamicLibrary());
      }
    }
    return libraryListBuilder.build();
  }

  public List<Artifact> getStaticModeParamsForDynamicLibraryLibraries() {
    ImmutableList.Builder<Artifact> artifactListBuilder = ImmutableList.builder();
    for (LibraryToLink library : getLibraries()) {
      if (library.getPicStaticLibrary() != null) {
        artifactListBuilder.add(library.getPicStaticLibrary());
      } else if (library.getStaticLibrary() != null) {
        artifactListBuilder.add(library.getStaticLibrary());
      } else if (library.getInterfaceLibrary() != null) {
        artifactListBuilder.add(library.getInterfaceLibrary());
      } else {
        artifactListBuilder.add(library.getDynamicLibrary());
      }
    }
    return artifactListBuilder.build();
  }

  public List<Artifact> getDynamicModeParamsForExecutableLibraries() {
    ImmutableList.Builder<Artifact> artifactListBuilder = ImmutableList.builder();
    for (LibraryToLink library : getLibraries()) {
      if (library.getInterfaceLibrary() != null) {
        artifactListBuilder.add(library.getInterfaceLibrary());
      } else if (library.getDynamicLibrary() != null) {
        artifactListBuilder.add(library.getDynamicLibrary());
      } else if (library.getStaticLibrary() != null) {
        artifactListBuilder.add(library.getStaticLibrary());
      } else if (library.getPicStaticLibrary() != null) {
        artifactListBuilder.add(library.getPicStaticLibrary());
      }
    }
    return artifactListBuilder.build();
  }

  public List<Artifact> getDynamicModeParamsForDynamicLibraryLibraries() {
    ImmutableList.Builder<Artifact> artifactListBuilder = ImmutableList.builder();
    for (LibraryToLink library : getLibraries()) {
      if (library.getInterfaceLibrary() != null) {
        artifactListBuilder.add(library.getInterfaceLibrary());
      } else if (library.getDynamicLibrary() != null) {
        artifactListBuilder.add(library.getDynamicLibrary());
      } else if (library.getPicStaticLibrary() != null) {
        artifactListBuilder.add(library.getPicStaticLibrary());
      } else if (library.getStaticLibrary() != null) {
        artifactListBuilder.add(library.getStaticLibrary());
      }
    }
    return artifactListBuilder.build();
  }

  public List<Artifact> getDynamicLibrariesForRuntime(boolean linkingStatically) {
    return LibraryToLink.getDynamicLibrariesForRuntime(linkingStatically, libraries);
  }

  public NestedSet<LibraryToLink> getLibraries() {
    return libraries;
  }

  @Override
  public SkylarkList<String> getSkylarkUserLinkFlags() {
    return SkylarkList.createImmutable(getFlattenedUserLinkFlags());
  }

  @Override
  public Object getSkylarkLibrariesToLink(StarlarkThread thread) {
    if (thread.getSemantics().incompatibleDepsetForLibrariesToLinkGetter()) {
      return SkylarkNestedSet.of(LibraryToLink.class, libraries);
    } else {
      return SkylarkList.createImmutable(libraries.toList());
    }
  }

  @Override
  public SkylarkNestedSet getSkylarkNonCodeInputs() {
    return SkylarkNestedSet.of(Artifact.class, nonCodeInputs);
  }

  public NestedSet<LinkOptions> getUserLinkFlags() {
    return userLinkFlags;
  }

  public ImmutableList<String> getFlattenedUserLinkFlags() {
    return Streams.stream(userLinkFlags)
        .map(LinkOptions::get)
        .flatMap(Collection::stream)
        .collect(ImmutableList.toImmutableList());
  }

  public NestedSet<Linkstamp> getLinkstamps() {
    return linkstamps;
  }

  public NestedSet<Artifact> getNonCodeInputs() {
    return nonCodeInputs;
  }

  public ExtraLinkTimeLibraries getExtraLinkTimeLibraries() {
    return extraLinkTimeLibraries;
  }

  public static Builder builder() {
    // private to avoid class initialization deadlock between this class and its outer class
    return new Builder();
  }

  /** Builder for {@link CcLinkingContext}. */
  public static class Builder {
    private final NestedSetBuilder<LibraryToLink> libraries = NestedSetBuilder.linkOrder();
    private final NestedSetBuilder<LinkOptions> userLinkFlags = NestedSetBuilder.linkOrder();
    private final NestedSetBuilder<Linkstamp> linkstamps = NestedSetBuilder.compileOrder();
    private final NestedSetBuilder<Artifact> nonCodeInputs = NestedSetBuilder.linkOrder();
    private ExtraLinkTimeLibraries extraLinkTimeLibraries = null;

    public Builder addLibraries(NestedSet<LibraryToLink> libraries) {
      this.libraries.addTransitive(libraries);
      return this;
    }

    public Builder addUserLinkFlags(NestedSet<LinkOptions> userLinkFlags) {
      this.userLinkFlags.addTransitive(userLinkFlags);
      return this;
    }

    Builder addLinkstamps(NestedSet<Linkstamp> linkstamps) {
      this.linkstamps.addTransitive(linkstamps);
      return this;
    }

    Builder addNonCodeInputs(NestedSet<Artifact> nonCodeInputs) {
      this.nonCodeInputs.addTransitive(nonCodeInputs);
      return this;
    }

    public Builder setExtraLinkTimeLibraries(ExtraLinkTimeLibraries extraLinkTimeLibraries) {
      Preconditions.checkState(this.extraLinkTimeLibraries == null);
      this.extraLinkTimeLibraries = extraLinkTimeLibraries;
      return this;
    }

    public CcLinkingContext build() {
      return new CcLinkingContext(
          libraries.build(),
          userLinkFlags.build(),
          linkstamps.build(),
          nonCodeInputs.build(),
          extraLinkTimeLibraries);
    }
  }

  @Override
  public boolean equals(Object otherObject) {
    if (!(otherObject instanceof CcLinkingContext)) {
      return false;
    }
    CcLinkingContext other = (CcLinkingContext) otherObject;
    if (this == other) {
      return true;
    }
    return this.libraries.shallowEquals(other.libraries)
        && this.userLinkFlags.shallowEquals(other.userLinkFlags)
        && this.linkstamps.shallowEquals(other.linkstamps)
        && this.nonCodeInputs.shallowEquals(other.nonCodeInputs);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(
        libraries.shallowHashCode(),
        userLinkFlags.shallowHashCode(),
        linkstamps.shallowHashCode(),
        nonCodeInputs.shallowHashCode());
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("userLinkFlags", userLinkFlags)
        .add("linkstamps", linkstamps)
        .add("libraries", libraries)
        .add("nonCodeInputs", nonCodeInputs)
        .toString();
  }
}
