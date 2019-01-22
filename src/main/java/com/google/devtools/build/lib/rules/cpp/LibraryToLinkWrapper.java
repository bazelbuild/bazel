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

import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.cpp.CcLinkParams.LinkOptions;
import com.google.devtools.build.lib.rules.cpp.CcLinkParams.Linkstamp;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.SolibLibraryToLink;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcLinkingContextApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.LibraryToLinkWrapperApi;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import java.util.Collection;
import java.util.List;
import java.util.ListIterator;
import javax.annotation.Nullable;

/**
 * Encapsulates information for linking a library.
 *
 * <p>TODO(b/118663806): We will be replacing {@link CcLinkParams} gradually as described in
 * b/118663806. This class which shall be renamed later to LibraryToLink (once the old LibraryToLink
 * implementation is removed) will have all the information necessary for linking a library in all
 * of its variants currently encapsulated in the four modes of {@link CcLinkParams} stored in {@link
 * CcLinkingInfo}, these modes are: static params for executable, static params for dynamic library,
 * dynamic params for executable and dynamic params for dynamic library.
 *
 * <p>To do this refactoring incrementally, we first introduce this class and add a method that is
 * able to convert from this representation to the old four CcLinkParams variables.
 */
public class LibraryToLinkWrapper implements LibraryToLinkWrapperApi {

  public static LibraryToLinkWrapper convertLinkOutputsToLibraryToLinkWrapper(
      CcLinkingOutputs ccLinkingOutputs) {
    Preconditions.checkState(!ccLinkingOutputs.isEmpty());

    Builder libraryToLinkWrapperBuilder = builder();
    if (!ccLinkingOutputs.getStaticLibraries().isEmpty()) {
      Preconditions.checkState(ccLinkingOutputs.getStaticLibraries().size() == 1);
      LibraryToLink staticLibrary = ccLinkingOutputs.getStaticLibraries().get(0);
      libraryToLinkWrapperBuilder.setStaticLibrary(staticLibrary.getArtifact());
      libraryToLinkWrapperBuilder.setObjectFiles(
          ImmutableList.copyOf(staticLibrary.getObjectFiles()));
      libraryToLinkWrapperBuilder.setLtoCompilationContext(
          staticLibrary.getLtoCompilationContext());
      libraryToLinkWrapperBuilder.setSharedNonLtoBackends(
          ImmutableMap.copyOf(staticLibrary.getSharedNonLtoBackends()));
      libraryToLinkWrapperBuilder.setAlwayslink(
          staticLibrary.getArtifactCategory() == ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY);
      libraryToLinkWrapperBuilder.setLibraryIdentifier(staticLibrary.getLibraryIdentifier());
    }

    if (!ccLinkingOutputs.getPicStaticLibraries().isEmpty()) {
      Preconditions.checkState(ccLinkingOutputs.getPicStaticLibraries().size() == 1);
      LibraryToLink picStaticLibrary = ccLinkingOutputs.getPicStaticLibraries().get(0);
      libraryToLinkWrapperBuilder.setPicStaticLibrary(picStaticLibrary.getArtifact());
      libraryToLinkWrapperBuilder.setPicObjectFiles(
          ImmutableList.copyOf(picStaticLibrary.getObjectFiles()));
      libraryToLinkWrapperBuilder.setPicLtoCompilationContext(
          picStaticLibrary.getLtoCompilationContext());
      libraryToLinkWrapperBuilder.setPicSharedNonLtoBackends(
          ImmutableMap.copyOf(picStaticLibrary.getSharedNonLtoBackends()));
      libraryToLinkWrapperBuilder.setAlwayslink(
          picStaticLibrary.getArtifactCategory() == ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY);
      libraryToLinkWrapperBuilder.setLibraryIdentifier(picStaticLibrary.getLibraryIdentifier());
    }

    if (!ccLinkingOutputs.getDynamicLibrariesForLinking().isEmpty()) {
      Preconditions.checkState(ccLinkingOutputs.getDynamicLibrariesForLinking().size() == 1);
      Preconditions.checkState(ccLinkingOutputs.getDynamicLibrariesForRuntime().size() == 1);
      LibraryToLink dynamicLibraryForLinking =
          ccLinkingOutputs.getDynamicLibrariesForLinking().get(0);
      LibraryToLink dynamicLibraryForRuntime =
          ccLinkingOutputs.getDynamicLibrariesForRuntime().get(0);
      if (dynamicLibraryForLinking != dynamicLibraryForRuntime) {
        libraryToLinkWrapperBuilder.setInterfaceLibrary(dynamicLibraryForLinking.getArtifact());
        if (dynamicLibraryForLinking instanceof SolibLibraryToLink) {
          libraryToLinkWrapperBuilder.setResolvedSymlinkInterfaceLibrary(
              dynamicLibraryForLinking.getOriginalLibraryArtifact());
        }
        libraryToLinkWrapperBuilder.setDynamicLibrary(dynamicLibraryForRuntime.getArtifact());
        if (dynamicLibraryForRuntime instanceof SolibLibraryToLink) {
          libraryToLinkWrapperBuilder.setResolvedSymlinkDynamicLibrary(
              dynamicLibraryForRuntime.getOriginalLibraryArtifact());
        }
      } else {
        libraryToLinkWrapperBuilder.setDynamicLibrary(dynamicLibraryForRuntime.getArtifact());
        if (dynamicLibraryForRuntime instanceof SolibLibraryToLink) {
          libraryToLinkWrapperBuilder.setResolvedSymlinkDynamicLibrary(
              dynamicLibraryForRuntime.getOriginalLibraryArtifact());
        }
      }
      libraryToLinkWrapperBuilder.setLibraryIdentifier(
          dynamicLibraryForLinking.getLibraryIdentifier());
    }
    return libraryToLinkWrapperBuilder.build();
  }

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
    if (dynamicLibrary == null) {
      return null;
    }
    if (linkingStatically && (staticLibrary != null || picStaticLibrary != null)) {
      return null;
    }
    return dynamicLibrary;
  }

  /** Structure of the new CcLinkingContext. This will replace {@link CcLinkingInfo}. */
  public static class CcLinkingContext implements CcLinkingContextApi {
    public static final CcLinkingContext EMPTY = CcLinkingContext.builder().build();

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
      ImmutableList.Builder<Artifact> dynamicLibrariesForRuntimeBuilder = ImmutableList.builder();
      for (LibraryToLinkWrapper libraryToLinkWrapper : libraries) {
        Artifact artifact =
            libraryToLinkWrapper.getDynamicLibraryForRuntimeOrNull(linkingStatically);
        if (artifact != null) {
          dynamicLibrariesForRuntimeBuilder.add(artifact);
        }
      }
      return dynamicLibrariesForRuntimeBuilder.build();
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

      public Builder addLinkstamps(NestedSet<Linkstamp> linkstamps) {
        this.linkstamps.addTransitive(linkstamps);
        return this;
      }

      public Builder addNonCodeInputs(NestedSet<Artifact> nonCodeInputs) {
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
      if (!this.libraries.shallowEquals(other.libraries)
          || !this.userLinkFlags.shallowEquals(other.userLinkFlags)
          || !this.linkstamps.shallowEquals(other.linkstamps)
          || !this.nonCodeInputs.shallowEquals(other.nonCodeInputs)) {
        return false;
      }
      return true;
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(
          libraries.shallowHashCode(),
          userLinkFlags.shallowHashCode(),
          linkstamps.shallowHashCode(),
          nonCodeInputs.shallowHashCode());
    }
  }

  private final String libraryIdentifier;

  private final Artifact staticLibrary;
  private final Iterable<Artifact> objectFiles;
  private final LtoCompilationContext ltoCompilationContext;
  private final ImmutableMap<Artifact, LtoBackendArtifacts> sharedNonLtoBackends;

  private final Artifact picStaticLibrary;
  private final Iterable<Artifact> picObjectFiles;
  private final LtoCompilationContext picLtoCompilationContext;
  private final ImmutableMap<Artifact, LtoBackendArtifacts> picSharedNonLtoBackends;

  private final Artifact dynamicLibrary;
  private final Artifact resolvedSymlinkDynamicLibrary;
  private final Artifact interfaceLibrary;
  private final Artifact resolvedSymlinkInterfaceLibrary;
  private final boolean alwayslink;

  private LibraryToLink picStaticLibraryToLink;
  private LibraryToLink staticLibraryToLink;
  private LibraryToLink dynamicLibraryToLink;
  private LibraryToLink interfaceLibraryToLink;

  // TODO(plf): This is just needed for Go, do not expose to Skylark and try to remove it. This was
  // introduced to let a linker input declare that it needs debug info in the executable.
  // Specifically, this was introduced for linking Go into a C++ binary when using the gccgo
  // compiler.
  boolean mustKeepDebug;

  private LibraryToLinkWrapper(
      String libraryIdentifier,
      Artifact staticLibrary,
      Iterable<Artifact> objectFiles,
      LtoCompilationContext ltoCompilationContext,
      ImmutableMap<Artifact, LtoBackendArtifacts> sharedNonLtoBackends,
      Artifact picStaticLibrary,
      Iterable<Artifact> picObjectFiles,
      LtoCompilationContext picLtoCompilationContext,
      ImmutableMap<Artifact, LtoBackendArtifacts> picSharedNonLtoBackends,
      Artifact dynamicLibrary,
      Artifact resolvedSymlinkDynamicLibrary,
      Artifact interfaceLibrary,
      Artifact resolvedSymlinkInterfaceLibrary,
      boolean alwayslink,
      boolean mustKeepDebug) {
    this.libraryIdentifier = libraryIdentifier;
    this.staticLibrary = staticLibrary;
    this.objectFiles = objectFiles;
    this.ltoCompilationContext = ltoCompilationContext;
    this.sharedNonLtoBackends = sharedNonLtoBackends;

    this.picStaticLibrary = picStaticLibrary;
    this.picObjectFiles = picObjectFiles;
    this.picLtoCompilationContext = picLtoCompilationContext;
    this.picSharedNonLtoBackends = picSharedNonLtoBackends;

    this.dynamicLibrary = dynamicLibrary;
    this.resolvedSymlinkDynamicLibrary = resolvedSymlinkDynamicLibrary;
    this.interfaceLibrary = interfaceLibrary;
    this.resolvedSymlinkInterfaceLibrary = resolvedSymlinkInterfaceLibrary;
    this.alwayslink = alwayslink;

    this.mustKeepDebug = mustKeepDebug;
  }

  public String getLibraryIdentifier() {
    return libraryIdentifier;
  }

  @Override
  public Artifact getStaticLibrary() {
    return staticLibrary;
  }

  public Iterable<Artifact> getObjectFiles() {
    return objectFiles;
  }

  public ImmutableMap<Artifact, LtoBackendArtifacts> getSharedNonLtoBackends() {
    return sharedNonLtoBackends;
  }

  public LtoCompilationContext getLtoCompilationContext() {
    return ltoCompilationContext;
  }

  @Override
  public Artifact getPicStaticLibrary() {
    return picStaticLibrary;
  }

  public Iterable<Artifact> getPicObjectFiles() {
    return picObjectFiles;
  }

  public ImmutableMap<Artifact, LtoBackendArtifacts> getPicSharedNonLtoBackends() {
    return picSharedNonLtoBackends;
  }

  public LtoCompilationContext getPicLtoCompilationContext() {
    return picLtoCompilationContext;
  }

  @Override
  public Artifact getDynamicLibrary() {
    return dynamicLibrary;
  }

  public Artifact getResolvedSymlinkDynamicLibrary() {
    return resolvedSymlinkDynamicLibrary;
  }

  @Override
  public Artifact getInterfaceLibrary() {
    return interfaceLibrary;
  }

  public Artifact getResolvedSymlinkInterfaceLibrary() {
    return resolvedSymlinkInterfaceLibrary;
  }

  @Override
  public boolean getAlwayslink() {
    return alwayslink;
  }

  public boolean getMustKeepDebug() {
    return mustKeepDebug;
  }

  public static Builder builder() {
    return new Builder();
  }

  @Nullable
  @SuppressWarnings("ReferenceEquality")
  public static String setDynamicArtifactsAndReturnIdentifier(
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
        // Note: with the old way of doing CcLinkParams, we lose the information regarding the
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

  public LibraryToLink getStaticLibraryToLink() {
    Preconditions.checkNotNull(staticLibrary);
    if (staticLibraryToLink != null) {
      return staticLibraryToLink;
    }
    staticLibraryToLink =
        LinkerInputs.newInputLibrary(
            staticLibrary,
            alwayslink
                ? ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY
                : ArtifactCategory.STATIC_LIBRARY,
            libraryIdentifier,
            objectFiles,
            ltoCompilationContext,
            sharedNonLtoBackends,
            mustKeepDebug);
    return staticLibraryToLink;
  }

  public LibraryToLink getPicStaticLibraryToLink() {
    Preconditions.checkNotNull(picStaticLibrary);
    if (picStaticLibraryToLink != null) {
      return picStaticLibraryToLink;
    }
    picStaticLibraryToLink =
        LinkerInputs.newInputLibrary(
            picStaticLibrary,
            alwayslink
                ? ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY
                : ArtifactCategory.STATIC_LIBRARY,
            libraryIdentifier,
            picObjectFiles,
            picLtoCompilationContext,
            picSharedNonLtoBackends,
            mustKeepDebug);
    return picStaticLibraryToLink;
  }

  public LibraryToLink getDynamicLibraryToLink() {
    Preconditions.checkNotNull(dynamicLibrary);
    if (dynamicLibraryToLink != null) {
      return dynamicLibraryToLink;
    }
    if (resolvedSymlinkDynamicLibrary != null) {
      dynamicLibraryToLink =
          LinkerInputs.solibLibraryToLink(
              dynamicLibrary, resolvedSymlinkDynamicLibrary, libraryIdentifier);
    } else {
      dynamicLibraryToLink =
          LinkerInputs.newInputLibrary(
              dynamicLibrary,
              ArtifactCategory.DYNAMIC_LIBRARY,
              libraryIdentifier,
              /* objectFiles */ ImmutableSet.of(),
              /* ltoCompilationContext */ new LtoCompilationContext(ImmutableMap.of()),
              /* sharedNonLtoBackends */ ImmutableMap.of(),
              mustKeepDebug);
    }
    return dynamicLibraryToLink;
  }

  public LibraryToLink getInterfaceLibraryToLink() {
    Preconditions.checkNotNull(interfaceLibrary);
    if (interfaceLibraryToLink != null) {
      return interfaceLibraryToLink;
    }
    if (resolvedSymlinkInterfaceLibrary != null) {
      interfaceLibraryToLink =
          LinkerInputs.solibLibraryToLink(
              interfaceLibrary, resolvedSymlinkInterfaceLibrary, libraryIdentifier);
    } else {
      interfaceLibraryToLink =
          LinkerInputs.newInputLibrary(
              interfaceLibrary,
              ArtifactCategory.INTERFACE_LIBRARY,
              libraryIdentifier,
              /* objectFiles */ ImmutableSet.of(),
              /* ltoCompilationContext */ new LtoCompilationContext(ImmutableMap.of()),
              /* sharedNonLtoBackends */ ImmutableMap.of(),
              mustKeepDebug);
    }
    return interfaceLibraryToLink;
  }

  /** Builder for LibraryToLinkWrapper. */
  public static class Builder {
    private String libraryIdentifier;

    private Artifact staticLibrary;
    private Iterable<Artifact> objectFiles;
    private LtoCompilationContext ltoCompilationContext;
    private ImmutableMap<Artifact, LtoBackendArtifacts> sharedNonLtoBackends;

    private Artifact picStaticLibrary;
    private Iterable<Artifact> picObjectFiles;
    private LtoCompilationContext picLtoCompilationContext;
    private ImmutableMap<Artifact, LtoBackendArtifacts> picSharedNonLtoBackends;

    private Artifact dynamicLibrary;
    private Artifact resolvedSymlinkDynamicLibrary;
    private Artifact interfaceLibrary;
    private Artifact resolvedSymlinkInterfaceLibrary;
    private boolean alwayslink;
    private boolean mustKeepDebug;

    private Builder() {}

    public Builder setLibraryIdentifier(String libraryIdentifier) {
      this.libraryIdentifier = libraryIdentifier;
      return this;
    }

    public Builder setStaticLibrary(Artifact staticLibrary) {
      this.staticLibrary = staticLibrary;
      return this;
    }

    public Builder setObjectFiles(Iterable<Artifact> objectFiles) {
      this.objectFiles = objectFiles;
      return this;
    }

    public Builder setLtoCompilationContext(LtoCompilationContext ltoCompilationContext) {
      this.ltoCompilationContext = ltoCompilationContext;
      return this;
    }

    public Builder setSharedNonLtoBackends(
        ImmutableMap<Artifact, LtoBackendArtifacts> sharedNonLtoBackends) {
      this.sharedNonLtoBackends = sharedNonLtoBackends;
      return this;
    }

    public Builder setPicStaticLibrary(Artifact picStaticLibrary) {
      this.picStaticLibrary = picStaticLibrary;
      return this;
    }

    public Builder setPicObjectFiles(Iterable<Artifact> picObjectFiles) {
      this.picObjectFiles = picObjectFiles;
      return this;
    }

    public Builder setPicLtoCompilationContext(LtoCompilationContext picLtoCompilationContext) {
      this.picLtoCompilationContext = picLtoCompilationContext;
      return this;
    }

    public Builder setPicSharedNonLtoBackends(
        ImmutableMap<Artifact, LtoBackendArtifacts> picSharedNonLtoBackends) {
      this.picSharedNonLtoBackends = picSharedNonLtoBackends;
      return this;
    }

    public Builder setDynamicLibrary(Artifact dynamicLibrary) {
      this.dynamicLibrary = dynamicLibrary;
      return this;
    }

    public Builder setResolvedSymlinkDynamicLibrary(Artifact resolvedSymlinkDynamicLibrary) {
      this.resolvedSymlinkDynamicLibrary = resolvedSymlinkDynamicLibrary;
      return this;
    }

    public Builder setInterfaceLibrary(Artifact interfaceLibrary) {
      this.interfaceLibrary = interfaceLibrary;
      return this;
    }

    public Builder setResolvedSymlinkInterfaceLibrary(Artifact resolvedSymlinkInterfaceLibrary) {
      this.resolvedSymlinkInterfaceLibrary = resolvedSymlinkInterfaceLibrary;
      return this;
    }

    public Builder setAlwayslink(boolean alwayslink) {
      this.alwayslink = alwayslink;
      return this;
    }

    public Builder setMustKeepDebug(boolean mustKeepDebug) {
      this.mustKeepDebug = mustKeepDebug;
      return this;
    }

    public LibraryToLinkWrapper build() {
      Preconditions.checkNotNull(libraryIdentifier);
      Preconditions.checkState(
          (objectFiles == null && ltoCompilationContext == null && sharedNonLtoBackends == null)
              || staticLibrary != null);
      Preconditions.checkState(
          (picObjectFiles == null
                  && picLtoCompilationContext == null
                  && picSharedNonLtoBackends == null)
              || picStaticLibrary != null);
      Preconditions.checkState(resolvedSymlinkDynamicLibrary == null || dynamicLibrary != null);
      Preconditions.checkState(resolvedSymlinkInterfaceLibrary == null || interfaceLibrary != null);
      Preconditions.checkState(
          staticLibrary != null
              || picStaticLibrary != null
              || dynamicLibrary != null
              || interfaceLibrary != null);

      return new LibraryToLinkWrapper(
          libraryIdentifier,
          staticLibrary,
          objectFiles,
          ltoCompilationContext,
          sharedNonLtoBackends,
          picStaticLibrary,
          picObjectFiles,
          picLtoCompilationContext,
          picSharedNonLtoBackends,
          dynamicLibrary,
          resolvedSymlinkDynamicLibrary,
          interfaceLibrary,
          resolvedSymlinkInterfaceLibrary,
          alwayslink,
          mustKeepDebug);
    }
  }
}
