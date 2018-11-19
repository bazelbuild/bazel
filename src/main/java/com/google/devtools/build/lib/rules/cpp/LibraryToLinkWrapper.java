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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.cpp.CcLinkParams.LinkOptions;
import com.google.devtools.build.lib.rules.cpp.CcLinkParams.Linkstamp;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.SolibLibraryToLink;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import java.util.Iterator;
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
public class LibraryToLinkWrapper {
  /** Structure of the new CcLinkingContext. This will replace {@link CcLinkingInfo}. */
  public static class CcLinkingContext {
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

    public NestedSet<LibraryToLinkWrapper> getLibraries() {
      return libraries;
    }

    public NestedSet<LinkOptions> getUserLinkFlags() {
      return userLinkFlags;
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

    public CcLinkingInfo toCcLinkingInfo() {
      return LibraryToLinkWrapper.toCcLinkingInfo(
          /* forcePic= */ false,
          ImmutableList.copyOf(libraries),
          userLinkFlags,
          linkstamps,
          nonCodeInputs,
          extraLinkTimeLibraries);
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
  }

  private final String libraryIdentifier;

  private final Artifact staticLibrary;
  private final Iterable<Artifact> objectFiles;
  private final ImmutableMap<Artifact, Artifact> ltoBitcodeFiles;
  private final ImmutableMap<Artifact, LtoBackendArtifacts> sharedNonLtoBackends;

  private final Artifact picStaticLibrary;
  private final Iterable<Artifact> picObjectFiles;
  private final ImmutableMap<Artifact, Artifact> picLtoBitcodeFiles;
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
      ImmutableMap<Artifact, Artifact> ltoBitcodeFiles,
      ImmutableMap<Artifact, LtoBackendArtifacts> sharedNonLtoBackends,
      Artifact picStaticLibrary,
      Iterable<Artifact> picObjectFiles,
      ImmutableMap<Artifact, Artifact> picLtoBitcodeFiles,
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
    this.ltoBitcodeFiles = ltoBitcodeFiles;
    this.sharedNonLtoBackends = sharedNonLtoBackends;

    this.picStaticLibrary = picStaticLibrary;
    this.picObjectFiles = picObjectFiles;
    this.picLtoBitcodeFiles = picLtoBitcodeFiles;
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

  public Artifact getStaticLibrary() {
    return staticLibrary;
  }

  public Iterable<Artifact> getObjectFiles() {
    return objectFiles;
  }

  public Artifact getPicStaticLibrary() {
    return picStaticLibrary;
  }

  public Artifact getDynamicLibrary() {
    return dynamicLibrary;
  }

  public Artifact getInterfaceLibrary() {
    return interfaceLibrary;
  }

  public boolean getAlwayslink() {
    return alwayslink;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static CcLinkingInfo toCcLinkingInfo(
      boolean forcePic,
      ImmutableCollection<LibraryToLinkWrapper> libraryToLinkWrappers,
      NestedSet<LinkOptions> linkOpts,
      NestedSet<Linkstamp> linkstamps,
      Iterable<Artifact> nonCodeInputs,
      ExtraLinkTimeLibraries extraLinkTimeLibraries) {
    CcLinkParams staticModeParamsForDynamicLibrary =
        buildStaticModeParamsForDynamicLibraryCcLinkParams(
            libraryToLinkWrappers, linkOpts, linkstamps, nonCodeInputs, extraLinkTimeLibraries);

    CcLinkParams staticModeParamsForExecutable;
    if (forcePic) {
      staticModeParamsForExecutable = staticModeParamsForDynamicLibrary;
    } else {
      staticModeParamsForExecutable =
          buildStaticModeParamsForExecutableCcLinkParams(
              libraryToLinkWrappers, linkOpts, linkstamps, nonCodeInputs, extraLinkTimeLibraries);
    }

    CcLinkParams dynamicModeParamsForDynamicLibrary =
        buildDynamicModeParamsForDynamicLibraryCcLinkParams(
            libraryToLinkWrappers, linkOpts, linkstamps, nonCodeInputs, extraLinkTimeLibraries);
    CcLinkParams dynamicModeParamsForExecutable;
    if (forcePic) {
      dynamicModeParamsForExecutable = dynamicModeParamsForDynamicLibrary;
    } else {
      dynamicModeParamsForExecutable =
          buildDynamicModeParamsForExecutableCcLinkParams(
              libraryToLinkWrappers, linkOpts, linkstamps, nonCodeInputs, extraLinkTimeLibraries);
    }

    CcLinkingInfo.Builder ccLinkingInfoBuilder =
        new CcLinkingInfo.Builder()
            .setStaticModeParamsForExecutable(staticModeParamsForExecutable)
            .setStaticModeParamsForDynamicLibrary(staticModeParamsForDynamicLibrary)
            .setDynamicModeParamsForExecutable(dynamicModeParamsForExecutable)
            .setDynamicModeParamsForDynamicLibrary(dynamicModeParamsForDynamicLibrary);
    return ccLinkingInfoBuilder.build();
  }

  /**
   * WARNING: If CcLinkingInfo contains linking parameters from transitive closure, this method will
   * be very expensive to execute because of nested set flattening. Should only be called by top
   * level targets that do transitive linking when the nested sets have to be flattened anyway.
   */
  public static CcLinkingContext fromCcLinkingInfo(CcLinkingInfo ccLinkingInfo) {
    CcLinkingContext.Builder ccLinkingContextBuilder = CcLinkingContext.builder();
    CcLinkParams staticModeParamsForExecutable = ccLinkingInfo.getStaticModeParamsForExecutable();
    CcLinkParams staticModeParamsForDynamicLibrary =
        ccLinkingInfo.getStaticModeParamsForDynamicLibrary();
    CcLinkParams dynamicModeParamsForExecutable = ccLinkingInfo.getDynamicModeParamsForExecutable();
    CcLinkParams dynamicModeParamsForDynamicLibrary =
        ccLinkingInfo.getDynamicModeParamsForDynamicLibrary();

    ccLinkingContextBuilder.addUserLinkFlags(
        getUserLinkFlags(
            staticModeParamsForExecutable,
            staticModeParamsForDynamicLibrary,
            dynamicModeParamsForExecutable,
            dynamicModeParamsForDynamicLibrary));

    ccLinkingContextBuilder.addLinkstamps(
        getLinkstamps(
            staticModeParamsForExecutable,
            staticModeParamsForDynamicLibrary,
            dynamicModeParamsForExecutable,
            dynamicModeParamsForDynamicLibrary));

    ccLinkingContextBuilder.addNonCodeInputs(
        getNonCodeInputs(
            staticModeParamsForExecutable,
            staticModeParamsForDynamicLibrary,
            dynamicModeParamsForExecutable,
            dynamicModeParamsForDynamicLibrary));

    ccLinkingContextBuilder.setExtraLinkTimeLibraries(
        getExtraLinkTimeLibraries(
            staticModeParamsForExecutable,
            staticModeParamsForDynamicLibrary,
            dynamicModeParamsForExecutable,
            dynamicModeParamsForDynamicLibrary));

    ccLinkingContextBuilder.addLibraries(
        getLibraries(
            staticModeParamsForExecutable,
            staticModeParamsForDynamicLibrary,
            dynamicModeParamsForExecutable,
            dynamicModeParamsForDynamicLibrary));
    return ccLinkingContextBuilder.build();
  }

  private static void checkAllSizesMatch(
      int staticModeForExecutable,
      int staticModeForDynamicLibrary,
      int dynamicModeForExecutable,
      int dynamicModeForDynamicLibrary) {
    Preconditions.checkState(
        staticModeForExecutable == staticModeForDynamicLibrary
            && staticModeForDynamicLibrary == dynamicModeForExecutable
            && dynamicModeForExecutable == dynamicModeForDynamicLibrary);
  }

  private static ExtraLinkTimeLibraries getExtraLinkTimeLibraries(
      CcLinkParams staticModeParamsForExecutable,
      CcLinkParams staticModeParamsForDynamicLibrary,
      CcLinkParams dynamicModeParamsForExecutable,
      CcLinkParams dynamicModeParamsForDynamicLibrary) {
    Preconditions.checkState(
        (staticModeParamsForExecutable.getExtraLinkTimeLibraries() == null
                && staticModeParamsForDynamicLibrary.getExtraLinkTimeLibraries() == null)
            || (staticModeParamsForExecutable.getExtraLinkTimeLibraries().getExtraLibraries().size()
                == staticModeParamsForDynamicLibrary
                    .getExtraLinkTimeLibraries()
                    .getExtraLibraries()
                    .size()));

    Preconditions.checkState(
        (staticModeParamsForDynamicLibrary.getExtraLinkTimeLibraries() == null
                && dynamicModeParamsForExecutable.getExtraLinkTimeLibraries() == null)
            || (staticModeParamsForDynamicLibrary
                    .getExtraLinkTimeLibraries()
                    .getExtraLibraries()
                    .size()
                == dynamicModeParamsForExecutable
                    .getExtraLinkTimeLibraries()
                    .getExtraLibraries()
                    .size()));
    Preconditions.checkState(
        (dynamicModeParamsForExecutable.getExtraLinkTimeLibraries() == null
                && dynamicModeParamsForDynamicLibrary.getExtraLinkTimeLibraries() == null)
            || (dynamicModeParamsForExecutable
                    .getExtraLinkTimeLibraries()
                    .getExtraLibraries()
                    .size()
                == dynamicModeParamsForDynamicLibrary
                    .getExtraLinkTimeLibraries()
                    .getExtraLibraries()
                    .size()));
    return staticModeParamsForExecutable.getExtraLinkTimeLibraries();
  }

  private static NestedSet<LinkOptions> getUserLinkFlags(
      CcLinkParams staticModeParamsForExecutable,
      CcLinkParams staticModeParamsForDynamicLibrary,
      CcLinkParams dynamicModeParamsForExecutable,
      CcLinkParams dynamicModeParamsForDynamicLibrary) {
    checkAllSizesMatch(
        staticModeParamsForExecutable.flattenedLinkopts().size(),
        staticModeParamsForDynamicLibrary.flattenedLinkopts().size(),
        dynamicModeParamsForExecutable.flattenedLinkopts().size(),
        dynamicModeParamsForDynamicLibrary.flattenedLinkopts().size());
    return staticModeParamsForExecutable.getLinkopts();
  }

  private static NestedSet<Linkstamp> getLinkstamps(
      CcLinkParams staticModeParamsForExecutable,
      CcLinkParams staticModeParamsForDynamicLibrary,
      CcLinkParams dynamicModeParamsForExecutable,
      CcLinkParams dynamicModeParamsForDynamicLibrary) {
    checkAllSizesMatch(
        staticModeParamsForExecutable.getLinkstamps().toList().size(),
        staticModeParamsForDynamicLibrary.getLinkstamps().toList().size(),
        dynamicModeParamsForExecutable.getLinkstamps().toList().size(),
        dynamicModeParamsForDynamicLibrary.getLinkstamps().toList().size());
    return staticModeParamsForExecutable.getLinkstamps();
  }

  private static NestedSet<Artifact> getNonCodeInputs(
      CcLinkParams staticModeParamsForExecutable,
      CcLinkParams staticModeParamsForDynamicLibrary,
      CcLinkParams dynamicModeParamsForExecutable,
      CcLinkParams dynamicModeParamsForDynamicLibrary) {
    checkAllSizesMatch(
        staticModeParamsForExecutable.getNonCodeInputs().toList().size(),
        staticModeParamsForDynamicLibrary.getNonCodeInputs().toList().size(),
        dynamicModeParamsForExecutable.getNonCodeInputs().toList().size(),
        dynamicModeParamsForDynamicLibrary.getNonCodeInputs().toList().size());
    return staticModeParamsForExecutable.getNonCodeInputs();
  }

  @Nullable
  private static String setStaticArtifactsAndReturnIdentifier(
      LibraryToLinkWrapper.Builder libraryToLinkWrapperBuilder,
      LibraryToLink staticModeParamsForExecutableEntry,
      LibraryToLink staticModeParamsForDynamicLibraryEntry) {
    LibraryToLinkByPicness noPicAndPicStaticLibraryToLink =
        returnNoPicAndPicStaticLibraryToLink(
            staticModeParamsForExecutableEntry, staticModeParamsForDynamicLibraryEntry);
    String libraryIdentifier = null;
    LibraryToLink noPicStaticLibrary = noPicAndPicStaticLibraryToLink.getNoPicLibrary();
    if (noPicStaticLibrary != null) {
      libraryToLinkWrapperBuilder.setStaticLibrary(noPicStaticLibrary.getArtifact());
      libraryIdentifier = noPicStaticLibrary.getLibraryIdentifier();
      if (noPicStaticLibrary.containsObjectFiles()) {
        libraryToLinkWrapperBuilder.setObjectFiles(noPicStaticLibrary.getObjectFiles());
      }
      libraryToLinkWrapperBuilder.setLtoBitcodeFiles(noPicStaticLibrary.getLtoBitcodeFiles());
      libraryToLinkWrapperBuilder.setSharedNonLtoBackends(
          noPicStaticLibrary.getSharedNonLtoBackends());
      libraryToLinkWrapperBuilder.setAlwayslink(
          noPicStaticLibrary.getArtifactCategory() == ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY);
    }
    LibraryToLink picStaticLibrary = noPicAndPicStaticLibraryToLink.getPicLibrary();
    if (picStaticLibrary != null) {
      libraryToLinkWrapperBuilder.setPicStaticLibrary(picStaticLibrary.getArtifact());
      if (libraryIdentifier == null) {
        libraryIdentifier = picStaticLibrary.getLibraryIdentifier();
      } else {
        Preconditions.checkState(libraryIdentifier.equals(picStaticLibrary.getLibraryIdentifier()));
      }
      if (picStaticLibrary.containsObjectFiles()) {
        libraryToLinkWrapperBuilder.setPicObjectFiles(picStaticLibrary.getObjectFiles());
      }
      libraryToLinkWrapperBuilder.setPicLtoBitcodeFiles(picStaticLibrary.getLtoBitcodeFiles());
      libraryToLinkWrapperBuilder.setPicSharedNonLtoBackends(
          picStaticLibrary.getSharedNonLtoBackends());
      libraryToLinkWrapperBuilder.setAlwayslink(
          picStaticLibrary.getArtifactCategory() == ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY);
    }

    return libraryIdentifier;
  }

  @Nullable
  private static String setDynamicArtifactsAndReturnIdentifier(
      LibraryToLinkWrapper.Builder libraryToLinkWrapperBuilder,
      LibraryToLink dynamicModeParamsForExecutableEntry,
      LibraryToLink dynamicModeParamsForDynamicLibraryEntry,
      Iterator<Artifact> runtimeLibraryIterator) {
    String libraryIdentifier = null;
    if (dynamicModeParamsForExecutableEntry.getArtifactCategory()
            == ArtifactCategory.DYNAMIC_LIBRARY
        || dynamicModeParamsForExecutableEntry.getArtifactCategory()
            == ArtifactCategory.INTERFACE_LIBRARY) {
      Preconditions.checkState(
          dynamicModeParamsForExecutableEntry == dynamicModeParamsForDynamicLibraryEntry);
      libraryIdentifier = dynamicModeParamsForExecutableEntry.getLibraryIdentifier();
    }

    if (dynamicModeParamsForExecutableEntry.getArtifactCategory()
        == ArtifactCategory.DYNAMIC_LIBRARY) {
      // The SolibLibraryToLink implementation returns ArtifactCategory.DYNAMIC_LIBRARY even if
      // the library being symlinked is an interface library. This was probably an oversight that
      // didn't cause any issues. In any case, here we have to figure out whether the library is
      // an interface library or not by checking the extension if it's a symlink.

      Preconditions.checkState(runtimeLibraryIterator.hasNext());
      Artifact runtimeArtifact = runtimeLibraryIterator.next();

      Artifact artifact = dynamicModeParamsForExecutableEntry.getArtifact();
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
          Preconditions.checkState(
              doArtifactsHaveSameBasename(
                  dynamicModeParamsForExecutableEntry.getArtifact(), runtimeArtifact));
          libraryToLinkWrapperBuilder.setDynamicLibrary(runtimeArtifact);
        } else {
          Preconditions.checkState(
              dynamicModeParamsForExecutableEntry.getArtifact() == runtimeArtifact);
          libraryToLinkWrapperBuilder.setDynamicLibrary(artifact);
          libraryToLinkWrapperBuilder.setResolvedSymlinkDynamicLibrary(
              dynamicModeParamsForExecutableEntry.getOriginalLibraryArtifact());
        }
      } else {
        libraryToLinkWrapperBuilder.setDynamicLibrary(artifact);
        Preconditions.checkState(
            dynamicModeParamsForExecutableEntry.getArtifact() == runtimeArtifact);
      }
    } else if (dynamicModeParamsForExecutableEntry.getArtifactCategory()
        == ArtifactCategory.INTERFACE_LIBRARY) {
      Preconditions.checkState(runtimeLibraryIterator.hasNext());
      Preconditions.checkState(
          !(dynamicModeParamsForExecutableEntry instanceof SolibLibraryToLink));
      Artifact runtimeArtifact = runtimeLibraryIterator.next();
      Preconditions.checkState(
          doArtifactsHaveSameBasename(
              dynamicModeParamsForExecutableEntry.getArtifact(), runtimeArtifact));
      libraryToLinkWrapperBuilder.setInterfaceLibrary(
          dynamicModeParamsForExecutableEntry.getArtifact());
      libraryToLinkWrapperBuilder.setDynamicLibrary(runtimeArtifact);
    }
    return libraryIdentifier;
  }

  private static NestedSet<LibraryToLinkWrapper> getLibraries(
      CcLinkParams staticModeParamsForExecutable,
      CcLinkParams staticModeParamsForDynamicLibrary,
      CcLinkParams dynamicModeParamsForExecutable,
      CcLinkParams dynamicModeParamsForDynamicLibrary) {
    Iterator<LibraryToLink> staticModeParamsForExecutableIterator =
        staticModeParamsForExecutable.getLibraries().iterator();
    Iterator<LibraryToLink> staticModeParamsForDynamicLibraryIterator =
        staticModeParamsForDynamicLibrary.getLibraries().iterator();
    Iterator<LibraryToLink> dynamicModeParamsForExecutableIterator =
        dynamicModeParamsForExecutable.getLibraries().iterator();
    Iterator<LibraryToLink> dynamicModeParamsForDynamicLibraryIterator =
        dynamicModeParamsForDynamicLibrary.getLibraries().iterator();

    Iterator<Artifact> runtimeLibraryIterator =
        dynamicModeParamsForExecutable.getDynamicLibrariesForRuntime().toList().iterator();

    NestedSetBuilder<LibraryToLinkWrapper> libraryToLinkWrappers = NestedSetBuilder.linkOrder();
    while (staticModeParamsForExecutableIterator.hasNext()
        && staticModeParamsForDynamicLibraryIterator.hasNext()
        && dynamicModeParamsForExecutableIterator.hasNext()
        && dynamicModeParamsForDynamicLibraryIterator.hasNext()) {
      LibraryToLinkWrapper.Builder libraryToLinkWrapperBuilder = LibraryToLinkWrapper.builder();
      LibraryToLink staticModeParamsForExecutableEntry =
          staticModeParamsForExecutableIterator.next();
      LibraryToLink staticModeParamsForDynamicLibraryEntry =
          staticModeParamsForDynamicLibraryIterator.next();

      String identifier =
          setStaticArtifactsAndReturnIdentifier(
              libraryToLinkWrapperBuilder,
              staticModeParamsForExecutableEntry,
              staticModeParamsForDynamicLibraryEntry);

      LibraryToLink dynamicModeParamsForExecutableEntry =
          dynamicModeParamsForExecutableIterator.next();
      LibraryToLink dynamicModeParamsForDynamicLibraryEntry =
          dynamicModeParamsForDynamicLibraryIterator.next();

      String dynamicLibraryIdentifier =
          setDynamicArtifactsAndReturnIdentifier(
              libraryToLinkWrapperBuilder,
              dynamicModeParamsForExecutableEntry,
              dynamicModeParamsForDynamicLibraryEntry,
              runtimeLibraryIterator);

      if (identifier == null) {
        identifier = dynamicLibraryIdentifier;
      } else {
        Preconditions.checkState(
            dynamicLibraryIdentifier == null || identifier.equals(dynamicLibraryIdentifier));
      }

      libraryToLinkWrapperBuilder.setLibraryIdentifier(identifier);

      Preconditions.checkState(
          staticModeParamsForExecutableEntry.isMustKeepDebug()
                  == staticModeParamsForDynamicLibraryEntry.isMustKeepDebug()
              && staticModeParamsForDynamicLibraryEntry.isMustKeepDebug()
                  == dynamicModeParamsForExecutableEntry.isMustKeepDebug()
              && dynamicModeParamsForExecutableEntry.isMustKeepDebug()
                  == dynamicModeParamsForDynamicLibraryEntry.isMustKeepDebug());
      libraryToLinkWrapperBuilder.setMustKeepDebug(
          staticModeParamsForExecutableEntry.isMustKeepDebug());

      libraryToLinkWrappers.add(libraryToLinkWrapperBuilder.build());
    }
    Preconditions.checkState(
        !(staticModeParamsForExecutableIterator.hasNext()
            || staticModeParamsForDynamicLibraryIterator.hasNext()
            || dynamicModeParamsForExecutableIterator.hasNext()
            || dynamicModeParamsForDynamicLibraryIterator.hasNext()));
    return libraryToLinkWrappers.build();
  }

  private static class LibraryToLinkByPicness {
    private final LibraryToLink noPicLibrary;
    private final LibraryToLink picLibrary;

    private LibraryToLinkByPicness(LibraryToLink noPicLibrary, LibraryToLink picLibrary) {
      this.noPicLibrary = noPicLibrary;
      this.picLibrary = picLibrary;
    }

    private LibraryToLink getNoPicLibrary() {
      return noPicLibrary;
    }

    private LibraryToLink getPicLibrary() {
      return picLibrary;
    }
  }

  /**
   * In the two static mode params objects of {@link CcLinkingInfo} we may have {@link
   * LibraryToLink} objects that are static libraries. This method grabs two instances, each coming
   * from one of the static mode params objects and returns a pair with a no-pic static
   * LibraryToLink as the first element and a pic static LibraryToLink in the second element.
   *
   * <p>We know that for dynamic libraries we will always prefer the pic variant, so if the
   * artifacts are different for the executable and for the dynamic library, then we know the former
   * is the no-pic static library and the latter is the pic static library.
   *
   * <p>If the artifacts are the same, then we check if they have the extension .pic. If they do,
   * then we know that there isn't a no-pic static library, so we return null for the first element.
   *
   * <p>If the artifacts are the same and they don't have the extension .pic. Then two of the
   * following things could be happening: 1. The static library is no-pic and the pic static library
   * wasn't generated. 2. The static library is pic and the no-pic static library wasn't generated.
   * This can only be happening if we created the static library from {@link CcCompilationHelper}.
   * When we create a static library from this class, the {@link LibraryToLink} will have the object
   * files used to create the library. We can look at the extension of these objects file to decide
   * if the library is pic or no-pic. If there are no object files, then the library must be no-pic.
   */
  private static LibraryToLinkByPicness returnNoPicAndPicStaticLibraryToLink(
      LibraryToLink fromStaticModeParamsForExecutable,
      LibraryToLink fromStaticModeParamsForDynamicLibrary) {
    if (fromStaticModeParamsForExecutable.getArtifactCategory() != ArtifactCategory.STATIC_LIBRARY
        && fromStaticModeParamsForExecutable.getArtifactCategory()
            != ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY) {
      return new LibraryToLinkByPicness(/* noPicLibrary= */ null, /* picLibrary= */ null);
    }
    Preconditions.checkState(
        fromStaticModeParamsForExecutable.getArtifactCategory()
            == fromStaticModeParamsForDynamicLibrary.getArtifactCategory());
    Artifact artifactFromStaticModeParamsForExecutable =
        fromStaticModeParamsForExecutable.getArtifact();
    Artifact artifactFromStaticModeParamsForDynamicLibrary =
        fromStaticModeParamsForDynamicLibrary.getArtifact();
    if (artifactFromStaticModeParamsForExecutable
        != artifactFromStaticModeParamsForDynamicLibrary) {
      Preconditions.checkState(
          !FileSystemUtils.removeExtension(artifactFromStaticModeParamsForExecutable.getFilename())
              .endsWith(".pic"));
      Preconditions.checkState(
          FileSystemUtils.removeExtension(
                  artifactFromStaticModeParamsForDynamicLibrary.getFilename())
              .endsWith(".pic"));
      return new LibraryToLinkByPicness(
          fromStaticModeParamsForExecutable, fromStaticModeParamsForDynamicLibrary);
    } else if (FileSystemUtils.removeExtension(
            artifactFromStaticModeParamsForExecutable.getFilename())
        .endsWith(".pic")) {
      return new LibraryToLinkByPicness(
          /* noPicLibrary= */ null, fromStaticModeParamsForDynamicLibrary);
    } else if (fromStaticModeParamsForExecutable.containsObjectFiles()
        && !Iterables.isEmpty(fromStaticModeParamsForExecutable.getObjectFiles())
        && FileSystemUtils.removeExtension(
                Iterables.getFirst(
                        fromStaticModeParamsForExecutable.getObjectFiles(),
                        /* defaultValue= */ null)
                    .getFilename())
            .endsWith(".pic")) {
      return new LibraryToLinkByPicness(
          /* noPicLibrary= */ null, fromStaticModeParamsForDynamicLibrary);
    }
    return new LibraryToLinkByPicness(
        fromStaticModeParamsForDynamicLibrary, /* picLibrary= */ null);
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

  /**
   * In this method and {@link #buildStaticModeParamsForDynamicLibraryCcLinkParams}, {@link
   * #buildDynamicModeParamsForExecutableCcLinkParams} and {@link
   * #buildDynamicModeParamsForDynamicLibraryCcLinkParams}, we add the ".a", ".pic.a" and/or ".so"
   * files in appropriate order of preference depending on the link preferences.
   *
   * <p>For static libraries, first choice is the PIC or no-PIC static variable, depending on
   * whether we prefer PIC or not. Even if we are using PIC, we still prefer the no PIC static
   * variant than using a dynamic library, although this may be an error later. Best performance is
   * obtained with no-PIC static libraries. If we don't have that we use the PIC variant, we can
   * live with the extra overhead.
   */
  private static CcLinkParams buildStaticModeParamsForExecutableCcLinkParams(
      ImmutableCollection<LibraryToLinkWrapper> libraryToLinkWrappers,
      NestedSet<LinkOptions> linkOpts,
      NestedSet<Linkstamp> linkstamps,
      Iterable<Artifact> nonCodeInputs,
      ExtraLinkTimeLibraries extraLinkTimeLibraries) {
    CcLinkParams.Builder ccLinkParamsBuilder =
        initializeCcLinkParams(linkOpts, linkstamps, nonCodeInputs, extraLinkTimeLibraries);
    for (LibraryToLinkWrapper libraryToLinkWrapper : libraryToLinkWrappers) {
      boolean usedDynamic = false;
      if (libraryToLinkWrapper.getStaticLibrary() != null) {
        ccLinkParamsBuilder.addLibrary(libraryToLinkWrapper.getStaticLibraryToLink());
      } else if (libraryToLinkWrapper.getPicStaticLibrary() != null) {
        ccLinkParamsBuilder.addLibrary(libraryToLinkWrapper.getPicStaticLibraryToLink());
      } else if (libraryToLinkWrapper.getInterfaceLibrary() != null) {
        usedDynamic = true;
        ccLinkParamsBuilder.addLibrary(libraryToLinkWrapper.getInterfaceLibraryToLink());
      } else if (libraryToLinkWrapper.getDynamicLibrary() != null) {
        usedDynamic = true;
        ccLinkParamsBuilder.addLibrary(libraryToLinkWrapper.getDynamicLibraryToLink());
      }

      if (usedDynamic && libraryToLinkWrapper.getDynamicLibrary() != null) {
        ccLinkParamsBuilder.addDynamicLibrariesForRuntime(
            ImmutableList.of(libraryToLinkWrapper.getDynamicLibrary()));
      }
    }
    return ccLinkParamsBuilder.build();
  }

  private static CcLinkParams buildStaticModeParamsForDynamicLibraryCcLinkParams(
      ImmutableCollection<LibraryToLinkWrapper> libraryToLinkWrappers,
      NestedSet<LinkOptions> linkOpts,
      NestedSet<Linkstamp> linkstamps,
      Iterable<Artifact> nonCodeInputs,
      ExtraLinkTimeLibraries extraLinkTimeLibraries) {
    CcLinkParams.Builder ccLinkParamsBuilder =
        initializeCcLinkParams(linkOpts, linkstamps, nonCodeInputs, extraLinkTimeLibraries);
    for (LibraryToLinkWrapper libraryToLinkWrapper : libraryToLinkWrappers) {
      boolean usedDynamic = false;
      if (libraryToLinkWrapper.getPicStaticLibrary() != null) {
        ccLinkParamsBuilder.addLibrary(libraryToLinkWrapper.getPicStaticLibraryToLink());
      } else if (libraryToLinkWrapper.getStaticLibrary() != null) {
        ccLinkParamsBuilder.addLibrary(libraryToLinkWrapper.getStaticLibraryToLink());
      } else if (libraryToLinkWrapper.getInterfaceLibrary() != null) {
        usedDynamic = true;
        ccLinkParamsBuilder.addLibrary(libraryToLinkWrapper.getInterfaceLibraryToLink());
      } else if (libraryToLinkWrapper.getDynamicLibrary() != null) {
        usedDynamic = true;
        ccLinkParamsBuilder.addLibrary(libraryToLinkWrapper.getDynamicLibraryToLink());
      }

      if (usedDynamic && libraryToLinkWrapper.getDynamicLibrary() != null) {
        ccLinkParamsBuilder.addDynamicLibrariesForRuntime(
            ImmutableList.of(libraryToLinkWrapper.getDynamicLibrary()));
      }
    }
    return ccLinkParamsBuilder.build();
  }

  private static CcLinkParams buildDynamicModeParamsForExecutableCcLinkParams(
      ImmutableCollection<LibraryToLinkWrapper> libraryToLinkWrappers,
      NestedSet<LinkOptions> linkOpts,
      NestedSet<Linkstamp> linkstamps,
      Iterable<Artifact> nonCodeInputs,
      ExtraLinkTimeLibraries extraLinkTimeLibraries) {
    CcLinkParams.Builder ccLinkParamsBuilder =
        initializeCcLinkParams(linkOpts, linkstamps, nonCodeInputs, extraLinkTimeLibraries);
    for (LibraryToLinkWrapper libraryToLinkWrapper : libraryToLinkWrappers) {
      boolean usedDynamic = false;
      if (libraryToLinkWrapper.getInterfaceLibrary() != null) {
        usedDynamic = true;
        ccLinkParamsBuilder.addLibrary(libraryToLinkWrapper.getInterfaceLibraryToLink());
      } else if (libraryToLinkWrapper.getDynamicLibrary() != null) {
        usedDynamic = true;
        ccLinkParamsBuilder.addLibrary(libraryToLinkWrapper.getDynamicLibraryToLink());
      } else if (libraryToLinkWrapper.getStaticLibrary() != null) {
        ccLinkParamsBuilder.addLibrary(libraryToLinkWrapper.getStaticLibraryToLink());
      } else if (libraryToLinkWrapper.getPicStaticLibrary() != null) {
        ccLinkParamsBuilder.addLibrary(libraryToLinkWrapper.getPicStaticLibraryToLink());
      }

      if (usedDynamic && libraryToLinkWrapper.getDynamicLibrary() != null) {
        ccLinkParamsBuilder.addDynamicLibrariesForRuntime(
            ImmutableList.of(libraryToLinkWrapper.getDynamicLibrary()));
      }
    }
    return ccLinkParamsBuilder.build();
  }

  private static CcLinkParams buildDynamicModeParamsForDynamicLibraryCcLinkParams(
      ImmutableCollection<LibraryToLinkWrapper> libraryToLinkWrappers,
      NestedSet<LinkOptions> linkOpts,
      NestedSet<Linkstamp> linkstamps,
      Iterable<Artifact> nonCodeInputs,
      ExtraLinkTimeLibraries extraLinkTimeLibraries) {
    CcLinkParams.Builder ccLinkParamsBuilder =
        initializeCcLinkParams(linkOpts, linkstamps, nonCodeInputs, extraLinkTimeLibraries);
    for (LibraryToLinkWrapper libraryToLinkWrapper : libraryToLinkWrappers) {
      boolean usedDynamic = false;
      if (libraryToLinkWrapper.getInterfaceLibrary() != null) {
        usedDynamic = true;
        ccLinkParamsBuilder.addLibrary(libraryToLinkWrapper.getInterfaceLibraryToLink());
      } else if (libraryToLinkWrapper.getDynamicLibrary() != null) {
        usedDynamic = true;
        ccLinkParamsBuilder.addLibrary(libraryToLinkWrapper.getDynamicLibraryToLink());
      } else if (libraryToLinkWrapper.getPicStaticLibrary() != null) {
        ccLinkParamsBuilder.addLibrary(libraryToLinkWrapper.getPicStaticLibraryToLink());
      } else if (libraryToLinkWrapper.getStaticLibrary() != null) {
        ccLinkParamsBuilder.addLibrary(libraryToLinkWrapper.getStaticLibraryToLink());
      }

      if (usedDynamic && libraryToLinkWrapper.getDynamicLibrary() != null) {
        ccLinkParamsBuilder.addDynamicLibrariesForRuntime(
            ImmutableList.of(libraryToLinkWrapper.getDynamicLibrary()));
      }
    }
    return ccLinkParamsBuilder.build();
  }

  private LibraryToLink getStaticLibraryToLink() {
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
            ltoBitcodeFiles,
            sharedNonLtoBackends,
            mustKeepDebug);
    return staticLibraryToLink;
  }

  private LibraryToLink getPicStaticLibraryToLink() {
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
            picLtoBitcodeFiles,
            picSharedNonLtoBackends,
            mustKeepDebug);
    return picStaticLibraryToLink;
  }

  private LibraryToLink getDynamicLibraryToLink() {
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
              /* ltoBitcodeFiles */ ImmutableMap.of(),
              /* sharedNonLtoBackends */ ImmutableMap.of(),
              mustKeepDebug);
    }
    return dynamicLibraryToLink;
  }

  private LibraryToLink getInterfaceLibraryToLink() {
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
              /* ltoBitcodeFiles */ ImmutableMap.of(),
              /* sharedNonLtoBackends */ ImmutableMap.of(),
              mustKeepDebug);
    }
    return interfaceLibraryToLink;
  }

  private static CcLinkParams.Builder initializeCcLinkParams(
      NestedSet<LinkOptions> linkOpts,
      NestedSet<Linkstamp> linkstamps,
      Iterable<Artifact> nonCodeInputs,
      ExtraLinkTimeLibraries extraLinkTimeLibraries) {
    CcLinkParams.Builder ccLinkParamsBuilder = CcLinkParams.builder();
    if (!linkOpts.isEmpty()) {
      ccLinkParamsBuilder.addLinkOpts(linkOpts);
    }
    ccLinkParamsBuilder.addLinkstamps(linkstamps);
    ccLinkParamsBuilder.addNonCodeInputs(nonCodeInputs);
    if (extraLinkTimeLibraries != null) {
      ccLinkParamsBuilder.addTransitiveExtraLinkTimeLibrary(extraLinkTimeLibraries);
    }
    return ccLinkParamsBuilder;
  }

  /** Builder for LibraryToLinkWrapper. */
  public static class Builder {
    private String libraryIdentifier;

    private Artifact staticLibrary;
    private Iterable<Artifact> objectFiles;
    private ImmutableMap<Artifact, Artifact> ltoBitcodeFiles;
    private ImmutableMap<Artifact, LtoBackendArtifacts> sharedNonLtoBackends;

    private Artifact picStaticLibrary;
    private Iterable<Artifact> picObjectFiles;
    private ImmutableMap<Artifact, Artifact> picLtoBitcodeFiles;
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

    public Builder setLtoBitcodeFiles(ImmutableMap<Artifact, Artifact> ltoBitcodeFiles) {
      this.ltoBitcodeFiles = ltoBitcodeFiles;
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

    public Builder setPicLtoBitcodeFiles(ImmutableMap<Artifact, Artifact> picLtoBitcodeFiles) {
      this.picLtoBitcodeFiles = picLtoBitcodeFiles;
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
          (objectFiles == null && ltoBitcodeFiles == null && sharedNonLtoBackends == null)
              || staticLibrary != null);
      Preconditions.checkState(
          (picObjectFiles == null && picLtoBitcodeFiles == null && picSharedNonLtoBackends == null)
              || picStaticLibrary != null);
      Preconditions.checkState(resolvedSymlinkDynamicLibrary == null || dynamicLibrary != null);
      Preconditions.checkState(resolvedSymlinkInterfaceLibrary == null || interfaceLibrary != null);
      Preconditions.checkState(!alwayslink || staticLibrary != null || picStaticLibrary != null);

      return new LibraryToLinkWrapper(
          libraryIdentifier,
          staticLibrary,
          objectFiles,
          ltoBitcodeFiles,
          sharedNonLtoBackends,
          picStaticLibrary,
          picObjectFiles,
          picLtoBitcodeFiles,
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
