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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import java.util.List;

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
      boolean alwayslink) {
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
      ImmutableCollection<String> linkOpts,
      NestedSet<Artifact> linkstamps,
      CcCompilationContext ccCompilationContext,
      Iterable<Artifact> nonCodeInputs) {
    CcLinkParams staticModeParamsForDynamicLibrary =
        buildStaticModeParamsForDynamicLibraryCcLinkParams(
            libraryToLinkWrappers, linkOpts, linkstamps, ccCompilationContext, nonCodeInputs);

    CcLinkParams staticModeParamsForExecutable;
    if (forcePic) {
      staticModeParamsForExecutable = staticModeParamsForDynamicLibrary;
    } else {
      staticModeParamsForExecutable =
          buildStaticModeParamsForExecutableCcLinkParams(
              libraryToLinkWrappers, linkOpts, linkstamps, ccCompilationContext, nonCodeInputs);
    }

    CcLinkParams dynamicModeParamsForDynamicLibrary =
        buildDynamicModeParamsForDynamicLibraryCcLinkParams(
            libraryToLinkWrappers, linkOpts, linkstamps, ccCompilationContext, nonCodeInputs);
    CcLinkParams dynamicModeParamsForExecutable;
    if (forcePic) {
      dynamicModeParamsForExecutable = dynamicModeParamsForDynamicLibrary;
    } else {
      dynamicModeParamsForExecutable =
          buildDynamicModeParamsForExecutableCcLinkParams(
              libraryToLinkWrappers, linkOpts, linkstamps, ccCompilationContext, nonCodeInputs);
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
      ImmutableCollection<String> linkOpts,
      NestedSet<Artifact> linkstamps,
      CcCompilationContext ccCompilationContext,
      Iterable<Artifact> nonCodeInputs) {
    CcLinkParams.Builder ccLinkParamsBuilder =
        initializeCcLinkParams(linkOpts, linkstamps, ccCompilationContext, nonCodeInputs);
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
      ImmutableCollection<String> linkOpts,
      NestedSet<Artifact> linkstamps,
      CcCompilationContext ccCompilationContext,
      Iterable<Artifact> nonCodeInputs) {
    CcLinkParams.Builder ccLinkParamsBuilder =
        initializeCcLinkParams(linkOpts, linkstamps, ccCompilationContext, nonCodeInputs);
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
      ImmutableCollection<String> linkOpts,
      NestedSet<Artifact> linkstamps,
      CcCompilationContext ccCompilationContext,
      Iterable<Artifact> nonCodeInputs) {
    CcLinkParams.Builder ccLinkParamsBuilder =
        initializeCcLinkParams(linkOpts, linkstamps, ccCompilationContext, nonCodeInputs);
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
      ImmutableCollection<String> linkOpts,
      NestedSet<Artifact> linkstamps,
      CcCompilationContext ccCompilationContext,
      Iterable<Artifact> nonCodeInputs) {
    CcLinkParams.Builder ccLinkParamsBuilder =
        initializeCcLinkParams(linkOpts, linkstamps, ccCompilationContext, nonCodeInputs);
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
            sharedNonLtoBackends);
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
            picSharedNonLtoBackends);
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
              /* sharedNonLtoBackends */ ImmutableMap.of());
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
              /* sharedNonLtoBackends */ ImmutableMap.of());
    }
    return interfaceLibraryToLink;
  }

  private static CcLinkParams.Builder initializeCcLinkParams(
      ImmutableCollection<String> linkOpts,
      NestedSet<Artifact> linkstamps,
      CcCompilationContext ccCompilationContext,
      Iterable<Artifact> nonCodeInputs) {
    CcLinkParams.Builder ccLinkParamsBuilder = CcLinkParams.builder();
    ccLinkParamsBuilder.addLinkOpts(linkOpts);
    ccLinkParamsBuilder.addLinkstamps(linkstamps, ccCompilationContext);
    ccLinkParamsBuilder.addNonCodeInputs(nonCodeInputs);
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

    private Builder() {}

    public Builder setLibraryIdentifier(String libraryIdentifier) {
      this.libraryIdentifier = libraryIdentifier;
      return this;
    }

    public Builder setStaticLibrary(Artifact staticLibrary) {
      this.staticLibrary = staticLibrary;
      return this;
    }

    public Builder setObjectFiles(List<Artifact> objectFiles) {
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

    public Builder setPicObjectFiles(List<Artifact> picObjectFiles) {
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
          alwayslink);
    }
  }
}
