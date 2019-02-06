package com.google.devtools.build.lib.rules.cpp;
// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoValue;
import com.google.common.base.Joiner;
import com.google.common.base.MoreObjects;
import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.skylark.SymbolGenerator;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.SolibLibraryToLink;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcLinkingContextApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.LibraryToLinkWrapperApi;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.ListIterator;
import javax.annotation.Nullable;

/**
 * Encapsulates information for linking a library.
 *
 * <p>TODO(b/118663806): This class which shall be renamed later to LibraryToLink (once the old
 * LibraryToLink implementation is removed) will have all the information necessary for linking a
 * library in all of its variants : static params for executable, static params for dynamic library,
 * dynamic params for executable and dynamic params for dynamic library.
 */
@AutoValue
public abstract class LibraryToLinkWrapper implements LibraryToLinkWrapperApi<Artifact> {

  public static List<LibraryToLink> convertLibraryToLinkWrapperListToLibraryToLinkList(
      NestedSet<LibraryToLinkWrapper> libraryToLinkWrappers,
      boolean staticMode,
      boolean forDynamicLibrary) {
    ImmutableList.Builder<LibraryToLink> librariesToLink = ImmutableList.builder();
    for (LibraryToLinkWrapper libraryToLinkWrapper : libraryToLinkWrappers) {
      LibraryToLink staticLibraryToLink =
          libraryToLinkWrapper.getStaticLibrary() == null
              ? null
              : libraryToLinkWrapper.getStaticLibraryToLink();
      LibraryToLink picStaticLibraryToLink =
          libraryToLinkWrapper.getPicStaticLibrary() == null
              ? null
              : libraryToLinkWrapper.getPicStaticLibraryToLink();
      LibraryToLink libraryToLinkToUse = null;
      if (staticMode) {
        if (forDynamicLibrary) {
          if (picStaticLibraryToLink != null) {
            libraryToLinkToUse = picStaticLibraryToLink;
          } else if (staticLibraryToLink != null) {
            libraryToLinkToUse = staticLibraryToLink;
          }
        } else {
          if (staticLibraryToLink != null) {
            libraryToLinkToUse = staticLibraryToLink;
          } else if (picStaticLibraryToLink != null) {
            libraryToLinkToUse = picStaticLibraryToLink;
          }
        }
        if (libraryToLinkToUse == null) {
          if (libraryToLinkWrapper.getInterfaceLibrary() != null) {
            libraryToLinkToUse = libraryToLinkWrapper.getInterfaceLibraryToLink();
          } else if (libraryToLinkWrapper.getDynamicLibrary() != null) {
            libraryToLinkToUse = libraryToLinkWrapper.getDynamicLibraryToLink();
          }
        }
      } else {
        if (libraryToLinkWrapper.getInterfaceLibrary() != null) {
          libraryToLinkToUse = libraryToLinkWrapper.getInterfaceLibraryToLink();
        } else if (libraryToLinkWrapper.getDynamicLibrary() != null) {
          libraryToLinkToUse = libraryToLinkWrapper.getDynamicLibraryToLink();
        }
        if (libraryToLinkToUse == null) {
          if (forDynamicLibrary) {
            if (picStaticLibraryToLink != null) {
              libraryToLinkToUse = picStaticLibraryToLink;
            } else if (staticLibraryToLink != null) {
              libraryToLinkToUse = staticLibraryToLink;
            }
          } else {
            if (staticLibraryToLink != null) {
              libraryToLinkToUse = staticLibraryToLink;
            } else if (picStaticLibraryToLink != null) {
              libraryToLinkToUse = picStaticLibraryToLink;
            }
          }
        }
      }
      Preconditions.checkNotNull(libraryToLinkToUse);
      librariesToLink.add(libraryToLinkToUse);
    }
    return librariesToLink.build();
  }

  public Artifact getDynamicLibraryForRuntimeOrNull(boolean linkingStatically) {
    if (getDynamicLibrary() == null) {
      return null;
    }
    if (linkingStatically && (getStaticLibrary() != null || getPicStaticLibrary() != null)) {
      return null;
    }
    return getDynamicLibrary();
  }

  /** Structure of CcLinkingContext. */
  public static class CcLinkingContext implements CcLinkingContextApi {
    public static final CcLinkingContext EMPTY = CcLinkingContext.builder().build();

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

      public static LinkOptions of(
          Iterable<String> linkOptions, SymbolGenerator<?> symbolGenerator) {
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
                "Unexpected inequality with equal symbols: " + this + ", " + that),
            ImmutableList.of());
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
     * <p>This object is required because linkstamp files may include other headers which will have
     * to be provided during compilation.
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

    private final NestedSet<LibraryToLinkWrapper> libraries;
    private final NestedSet<LinkOptions> userLinkFlags;
    private final NestedSet<Linkstamp> linkstamps;
    private final NestedSet<Artifact> nonCodeInputs;
    private final ExtraLinkTimeLibraries extraLinkTimeLibraries;

    public CcLinkingContext(
        NestedSet<LibraryToLinkWrapper> libraries,
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
      CcLinkingContext.Builder mergedCcLinkingContext = CcLinkingContext.builder();
      ExtraLinkTimeLibraries.Builder mergedExtraLinkTimeLibraries =
          ExtraLinkTimeLibraries.builder();
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
      for (LibraryToLinkWrapper libraryToLinkWrapper : getLibraries()) {
        if (libraryToLinkWrapper.getStaticLibrary() != null) {
          libraryListBuilder.add(libraryToLinkWrapper.getStaticLibrary());
        } else if (libraryToLinkWrapper.getPicStaticLibrary() != null) {
          libraryListBuilder.add(libraryToLinkWrapper.getPicStaticLibrary());
        } else if (libraryToLinkWrapper.getInterfaceLibrary() != null) {
          libraryListBuilder.add(libraryToLinkWrapper.getInterfaceLibrary());
        } else {
          libraryListBuilder.add(libraryToLinkWrapper.getDynamicLibrary());
        }
      }
      return libraryListBuilder.build();
    }

    public List<Artifact> getStaticModeParamsForDynamicLibraryLibraries() {
      ImmutableList.Builder<Artifact> artifactListBuilder = ImmutableList.builder();
      for (LibraryToLinkWrapper library : getLibraries()) {
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
      for (LibraryToLinkWrapper library : getLibraries()) {
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
      for (LibraryToLinkWrapper library : getLibraries()) {
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
      return LibraryToLinkWrapper.getDynamicLibrariesForRuntime(linkingStatically, libraries);
    }

    public NestedSet<LibraryToLinkWrapper> getLibraries() {
      return libraries;
    }

    @Override
    public SkylarkList<String> getSkylarkUserLinkFlags() {
      return SkylarkList.createImmutable(getFlattenedUserLinkFlags());
    }

    @Override
    public SkylarkList<LibraryToLinkWrapperApi> getSkylarkLibrariesToLink() {
      return SkylarkList.createImmutable(libraries.toList());
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
      return new Builder();
    }

    /** Builder for {@link CcLinkingContext}. */
    public static class Builder {
      private final NestedSetBuilder<LibraryToLinkWrapper> libraries = NestedSetBuilder.linkOrder();
      private final NestedSetBuilder<LinkOptions> userLinkFlags = NestedSetBuilder.linkOrder();
      private final NestedSetBuilder<Linkstamp> linkstamps = NestedSetBuilder.compileOrder();
      private final NestedSetBuilder<Artifact> nonCodeInputs = NestedSetBuilder.linkOrder();
      private ExtraLinkTimeLibraries extraLinkTimeLibraries = null;

      public Builder addLibraries(NestedSet<LibraryToLinkWrapper> libraries) {
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

  private LibraryToLink picStaticLibraryToLink;
  private LibraryToLink staticLibraryToLink;
  private LibraryToLink dynamicLibraryToLink;
  private LibraryToLink interfaceLibraryToLink;

  public abstract String getLibraryIdentifier();

  @Nullable
  @Override
  public abstract Artifact getStaticLibrary();

  @Nullable
  public abstract ImmutableList<Artifact> getObjectFiles();

  @Nullable
  public abstract ImmutableMap<Artifact, LtoBackendArtifacts> getSharedNonLtoBackends();

  @Nullable
  public abstract LtoCompilationContext getLtoCompilationContext();

  @Nullable
  @Override
  public abstract Artifact getPicStaticLibrary();

  @Nullable
  public abstract ImmutableList<Artifact> getPicObjectFiles();

  @Nullable
  public abstract ImmutableMap<Artifact, LtoBackendArtifacts> getPicSharedNonLtoBackends();

  @Nullable
  public abstract LtoCompilationContext getPicLtoCompilationContext();

  @Nullable
  @Override
  public abstract Artifact getDynamicLibrary();

  @Nullable
  public abstract Artifact getResolvedSymlinkDynamicLibrary();

  @Nullable
  @Override
  public abstract Artifact getInterfaceLibrary();

  @Nullable
  public abstract Artifact getResolvedSymlinkInterfaceLibrary();

  @Override
  public abstract boolean getAlwayslink();

  // TODO(plf): This is just needed for Go, do not expose to Skylark and try to remove it. This was
  // introduced to let a linker input declare that it needs debug info in the executable.
  // Specifically, this was introduced for linking Go into a C++ binary when using the gccgo
  // compiler.
  abstract boolean getMustKeepDebug();

  public static Builder builder() {
    return new AutoValue_LibraryToLinkWrapper.Builder()
        .setMustKeepDebug(false)
        .setAlwayslink(false);
  }

  @Nullable
  @SuppressWarnings("ReferenceEquality")
  static String setDynamicArtifactsAndReturnIdentifier(
      LibraryToLinkWrapper.Builder libraryToLinkWrapperBuilder,
      LibraryToLink dynamicModeParamsForExecutableEntry,
      LibraryToLink dynamicModeParamsForDynamicLibraryEntry,
      ListIterator<Artifact> runtimeLibraryIterator) {
    Preconditions.checkNotNull(runtimeLibraryIterator);
    Artifact artifact = dynamicModeParamsForExecutableEntry.getArtifact();
    String libraryIdentifier = null;
    Artifact runtimeArtifact = null;
    if (dynamicModeParamsForExecutableEntry.getArtifactCategory()
            == ArtifactCategory.DYNAMIC_LIBRARY
        || dynamicModeParamsForExecutableEntry.getArtifactCategory()
            == ArtifactCategory.INTERFACE_LIBRARY) {
      Preconditions.checkState(
          dynamicModeParamsForExecutableEntry == dynamicModeParamsForDynamicLibraryEntry);
      libraryIdentifier = dynamicModeParamsForExecutableEntry.getLibraryIdentifier();

      // Not every library to link has a corresponding runtime artifact, for example this is the
      // case when it is provided by the system. Here we check if the next runtime artifact has the
      // same basename as the current library to link, if it is, then we match them together. If
      // isn't, then we must rewind the iterator since every call to next() advances it.
      if (runtimeLibraryIterator.hasNext()) {
        runtimeArtifact = runtimeLibraryIterator.next();
        if (!doArtifactsHaveSameBasename(artifact, runtimeArtifact)) {
          runtimeArtifact = null;
          runtimeLibraryIterator.previous();
        }
      }
    }

    if (dynamicModeParamsForExecutableEntry.getArtifactCategory()
        == ArtifactCategory.DYNAMIC_LIBRARY) {
      // The SolibLibraryToLink implementation returns ArtifactCategory.DYNAMIC_LIBRARY even if
      // the library being symlinked is an interface library. This was probably an oversight that
      // didn't cause any issues. In any case, here we have to figure out whether the library is
      // an interface library or not by checking the extension if it's a symlink.

      if (dynamicModeParamsForExecutableEntry instanceof SolibLibraryToLink) {
        // Note: with the old way of doing C++ linking, we lose the information regarding the
        // runtime library. If {@code runtimeArtifact} is a symlink we only have the symlink but
        // not a reference to the artifact it points to. We can infer whether it's a symlink by
        // looking at whether the interface library is a symlink, however, we can't find out what
        // it points to. With the new API design, we won't lose this information anymore. This is
        // the way it has been done until now, but it wasn't a problem because symlinks get
        // automatically resolved when they are in runfiles.
        if (ArtifactCategory.INTERFACE_LIBRARY
            .getAllowedExtensions()
            .contains("." + artifact.getExtension())) {
          libraryToLinkWrapperBuilder.setInterfaceLibrary(artifact);
          libraryToLinkWrapperBuilder.setResolvedSymlinkInterfaceLibrary(
              dynamicModeParamsForExecutableEntry.getOriginalLibraryArtifact());
          if (runtimeArtifact != null) {
            libraryToLinkWrapperBuilder.setDynamicLibrary(runtimeArtifact);
          }
        } else {
          Preconditions.checkState(runtimeArtifact == null || artifact == runtimeArtifact);
          libraryToLinkWrapperBuilder.setDynamicLibrary(artifact);
          libraryToLinkWrapperBuilder.setResolvedSymlinkDynamicLibrary(
              dynamicModeParamsForExecutableEntry.getOriginalLibraryArtifact());
        }
      } else {
        libraryToLinkWrapperBuilder.setDynamicLibrary(artifact);
        Preconditions.checkState(runtimeArtifact == null || artifact == runtimeArtifact);
      }
    } else if (dynamicModeParamsForExecutableEntry.getArtifactCategory()
        == ArtifactCategory.INTERFACE_LIBRARY) {
      Preconditions.checkState(
          !(dynamicModeParamsForExecutableEntry instanceof SolibLibraryToLink));
      libraryToLinkWrapperBuilder.setInterfaceLibrary(artifact);
      if (runtimeArtifact != null) {
        libraryToLinkWrapperBuilder.setDynamicLibrary(runtimeArtifact);
      }
    }
    return libraryIdentifier;
  }

  private static boolean doArtifactsHaveSameBasename(Artifact first, Artifact second) {
    String nameFirst = removeAllExtensions(first.getRootRelativePath().getPathString());
    String nameSecond = removeAllExtensions(second.getRootRelativePath().getPathString());
    return nameFirst.equals(nameSecond);
  }

  private static String removeAllExtensions(String name) {
    String previousWithoutExtension = FileSystemUtils.removeExtension(name);
    String currentWithoutExtension = FileSystemUtils.removeExtension(previousWithoutExtension);
    while (!previousWithoutExtension.equals(currentWithoutExtension)) {
      previousWithoutExtension = currentWithoutExtension;
      currentWithoutExtension = FileSystemUtils.removeExtension(previousWithoutExtension);
    }
    return currentWithoutExtension;
  }

  LibraryToLink getStaticLibraryToLink() {
    Preconditions.checkNotNull(getStaticLibrary(), this);
    if (staticLibraryToLink != null) {
      return staticLibraryToLink;
    }
    staticLibraryToLink =
        LinkerInputs.newInputLibrary(
            getStaticLibrary(),
            getAlwayslink()
                ? ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY
                : ArtifactCategory.STATIC_LIBRARY,
            getLibraryIdentifier(),
            getObjectFiles(),
            getLtoCompilationContext(),
            getSharedNonLtoBackends(),
            getMustKeepDebug());
    return staticLibraryToLink;
  }

  LibraryToLink getPicStaticLibraryToLink() {
    Preconditions.checkNotNull(getPicStaticLibrary(), this);
    if (picStaticLibraryToLink != null) {
      return picStaticLibraryToLink;
    }
    picStaticLibraryToLink =
        LinkerInputs.newInputLibrary(
            getPicStaticLibrary(),
            getAlwayslink()
                ? ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY
                : ArtifactCategory.STATIC_LIBRARY,
            getLibraryIdentifier(),
            getPicObjectFiles(),
            getPicLtoCompilationContext(),
            getPicSharedNonLtoBackends(),
            getMustKeepDebug());
    return picStaticLibraryToLink;
  }

  LibraryToLink getDynamicLibraryToLink() {
    Preconditions.checkNotNull(getDynamicLibrary(), this);
    if (dynamicLibraryToLink != null) {
      return dynamicLibraryToLink;
    }
    if (getResolvedSymlinkDynamicLibrary() != null) {
      dynamicLibraryToLink =
          LinkerInputs.solibLibraryToLink(
              getDynamicLibrary(), getResolvedSymlinkDynamicLibrary(), getLibraryIdentifier());
    } else {
      dynamicLibraryToLink =
          LinkerInputs.newInputLibrary(
              getDynamicLibrary(),
              ArtifactCategory.DYNAMIC_LIBRARY,
              getLibraryIdentifier(),
              /* objectFiles */ ImmutableSet.of(),
              LtoCompilationContext.EMPTY,
              /* sharedNonLtoBackends */ ImmutableMap.of(),
              getMustKeepDebug());
    }
    return dynamicLibraryToLink;
  }

  LibraryToLink getInterfaceLibraryToLink() {
    Preconditions.checkNotNull(getInterfaceLibrary());
    if (interfaceLibraryToLink != null) {
      return interfaceLibraryToLink;
    }
    if (getResolvedSymlinkInterfaceLibrary() != null) {
      interfaceLibraryToLink =
          LinkerInputs.solibLibraryToLink(
              getInterfaceLibrary(), getResolvedSymlinkInterfaceLibrary(), getLibraryIdentifier());
    } else {
      interfaceLibraryToLink =
          LinkerInputs.newInputLibrary(
              getInterfaceLibrary(),
              ArtifactCategory.INTERFACE_LIBRARY,
              getLibraryIdentifier(),
              /* objectFiles */ ImmutableSet.of(),
              LtoCompilationContext.EMPTY,
              /* sharedNonLtoBackends */ ImmutableMap.of(),
              getMustKeepDebug());
    }
    return interfaceLibraryToLink;
  }

  public static List<Artifact> getDynamicLibrariesForRuntime(
      boolean linkingStatically, Iterable<LibraryToLinkWrapper> libraries) {
    ImmutableList.Builder<Artifact> dynamicLibrariesForRuntimeBuilder = ImmutableList.builder();
    for (LibraryToLinkWrapper libraryToLinkWrapper : libraries) {
      Artifact artifact = libraryToLinkWrapper.getDynamicLibraryForRuntimeOrNull(linkingStatically);
      if (artifact != null) {
        dynamicLibrariesForRuntimeBuilder.add(artifact);
      }
    }
    return dynamicLibrariesForRuntimeBuilder.build();
  }

  public static List<Artifact> getDynamicLibrariesForLinking(
      Iterable<LibraryToLinkWrapper> libraries) {
    ImmutableList.Builder<Artifact> dynamicLibrariesForLinkingBuilder = ImmutableList.builder();
    for (LibraryToLinkWrapper libraryToLinkWrapper : libraries) {
      if (libraryToLinkWrapper.getInterfaceLibrary() != null) {
        dynamicLibrariesForLinkingBuilder.add(libraryToLinkWrapper.getInterfaceLibrary());
      } else if (libraryToLinkWrapper.getDynamicLibrary() != null) {
        dynamicLibrariesForLinkingBuilder.add(libraryToLinkWrapper.getDynamicLibrary());
      }
    }
    return dynamicLibrariesForLinkingBuilder.build();
  }

  /** Builder for LibraryToLinkWrapper. */
  @AutoValue.Builder
  public abstract static class Builder {

    public abstract Builder setLibraryIdentifier(String libraryIdentifier);

    public abstract Builder setStaticLibrary(Artifact staticLibrary);

    public abstract Builder setObjectFiles(ImmutableList<Artifact> objectFiles);

    abstract Builder setLtoCompilationContext(LtoCompilationContext ltoCompilationContext);

    abstract Builder setSharedNonLtoBackends(
        ImmutableMap<Artifact, LtoBackendArtifacts> sharedNonLtoBackends);

    abstract Builder setPicStaticLibrary(Artifact picStaticLibrary);

    abstract Builder setPicObjectFiles(ImmutableList<Artifact> picObjectFiles);

    abstract Builder setPicLtoCompilationContext(LtoCompilationContext picLtoCompilationContext);

    abstract Builder setPicSharedNonLtoBackends(
        ImmutableMap<Artifact, LtoBackendArtifacts> picSharedNonLtoBackends);

    public abstract Builder setDynamicLibrary(Artifact dynamicLibrary);

    abstract Builder setResolvedSymlinkDynamicLibrary(Artifact resolvedSymlinkDynamicLibrary);

    abstract Builder setInterfaceLibrary(Artifact interfaceLibrary);

    abstract Builder setResolvedSymlinkInterfaceLibrary(Artifact resolvedSymlinkInterfaceLibrary);

    public abstract Builder setAlwayslink(boolean alwayslink);

    public abstract Builder setMustKeepDebug(boolean mustKeepDebug);

    // Methods just for validation, not to be called externally.
    abstract LibraryToLinkWrapper autoBuild();

    abstract String getLibraryIdentifier();

    abstract Artifact getStaticLibrary();

    abstract ImmutableList<Artifact> getObjectFiles();

    abstract ImmutableMap<Artifact, LtoBackendArtifacts> getSharedNonLtoBackends();

    abstract LtoCompilationContext getLtoCompilationContext();

    abstract Artifact getPicStaticLibrary();

    abstract ImmutableList<Artifact> getPicObjectFiles();

    abstract ImmutableMap<Artifact, LtoBackendArtifacts> getPicSharedNonLtoBackends();

    abstract LtoCompilationContext getPicLtoCompilationContext();

    abstract Artifact getDynamicLibrary();

    abstract Artifact getResolvedSymlinkDynamicLibrary();

    abstract Artifact getInterfaceLibrary();

    abstract Artifact getResolvedSymlinkInterfaceLibrary();

    public LibraryToLinkWrapper build() {
      Preconditions.checkNotNull(getLibraryIdentifier());
      Preconditions.checkState(
          (getObjectFiles() == null
                  && getLtoCompilationContext() == null
                  && getSharedNonLtoBackends() == null)
              || getStaticLibrary() != null);
      Preconditions.checkState(
          (getPicObjectFiles() == null
                  && getPicLtoCompilationContext() == null
                  && getPicSharedNonLtoBackends() == null)
              || getPicStaticLibrary() != null);
      Preconditions.checkState(
          getResolvedSymlinkDynamicLibrary() == null || getDynamicLibrary() != null);
      Preconditions.checkState(
          getResolvedSymlinkInterfaceLibrary() == null
              || getResolvedSymlinkInterfaceLibrary() != null);
      Preconditions.checkState(
          getStaticLibrary() != null
              || getPicStaticLibrary() != null
              || getDynamicLibrary() != null
              || getInterfaceLibrary() != null);

      return autoBuild();
    }
  }
}

