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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.LibraryToLinkApi;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;

/**
 * Encapsulates information for linking a library.
 *
 * <p>TODO(b/118663806): This class which shall be renamed later to LibraryToLink (once the old
 * LibraryToLink implementation is removed) will have all the information necessary for linking a
 * library in all of its variants : static params for executable, static params for dynamic library,
 * dynamic params for executable and dynamic params for dynamic library.
 */
@AutoValue
@Immutable
public abstract class LibraryToLink implements LibraryToLinkApi<Artifact, LtoBackendArtifacts> {

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  public static final Depset.ElementType TYPE = Depset.ElementType.of(LibraryToLink.class);

  public Artifact getDynamicLibraryForRuntimeOrNull(boolean linkingStatically) {
    if (getDynamicLibrary() == null) {
      return null;
    }
    if (linkingStatically && (getStaticLibrary() != null || getPicStaticLibrary() != null)) {
      return null;
    }
    return getDynamicLibrary();
  }

  private LinkerInputs.LibraryToLink picStaticLibraryToLink;
  private LinkerInputs.LibraryToLink staticLibraryToLink;
  private LinkerInputs.LibraryToLink dynamicLibraryToLink;
  private LinkerInputs.LibraryToLink interfaceLibraryToLink;

  public abstract String getLibraryIdentifier();

  @Nullable
  @Override
  public abstract Artifact getStaticLibrary();

  @Nullable
  public abstract ImmutableList<Artifact> getObjectFiles();

  @Nullable
  @Override
  public Sequence<Artifact> getObjectFilesForStarlark() {
    if (getObjectFiles() == null) {
      return StarlarkList.empty();
    }
    return StarlarkList.immutableCopyOf(getObjectFiles());
  }

  @Override
  public boolean getMustKeepDebugForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return getMustKeepDebug();
  }

  @Nullable
  @Override
  public Sequence<Artifact> getLtoBitcodeFilesForStarlark() {
    if (getLtoCompilationContext() == null) {
      return StarlarkList.empty();
    }
    return StarlarkList.immutableCopyOf(getLtoCompilationContext().getBitcodeFiles());
  }

  @Nullable
  public abstract ImmutableMap<Artifact, LtoBackendArtifacts> getSharedNonLtoBackends();

  @Nullable
  @Override
  public Dict<Artifact, LtoBackendArtifacts> getSharedNonLtoBackendsForStarlark(
      StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return Dict.immutableCopyOf(getSharedNonLtoBackends());
  }

  @Nullable
  public abstract LtoCompilationContext getLtoCompilationContext();

  @Nullable
  @Override
  public abstract Artifact getPicStaticLibrary();

  @Nullable
  public abstract ImmutableList<Artifact> getPicObjectFiles();

  @Nullable
  @Override
  public Sequence<Artifact> getPicObjectFilesForStarlark() {
    if (getPicObjectFiles() == null) {
      return StarlarkList.empty();
    }
    return StarlarkList.immutableCopyOf(getPicObjectFiles());
  }

  @Nullable
  @Override
  public Sequence<Artifact> getPicLtoBitcodeFilesForStarlark() {
    if (getPicLtoCompilationContext() == null) {
      return StarlarkList.empty();
    }
    return StarlarkList.immutableCopyOf(getPicLtoCompilationContext().getBitcodeFiles());
  }

  @Nullable
  public abstract ImmutableMap<Artifact, LtoBackendArtifacts> getPicSharedNonLtoBackends();

  @Nullable
  @Override
  public Dict<Artifact, LtoBackendArtifacts> getPicSharedNonLtoBackendsForStarlark(
      StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return Dict.immutableCopyOf(getPicSharedNonLtoBackends());
  }

  @Nullable
  public abstract LtoCompilationContext getPicLtoCompilationContext();

  @Nullable
  @Override
  public abstract Artifact getDynamicLibrary();

  @Nullable
  @Override
  public abstract Artifact getResolvedSymlinkDynamicLibrary();

  @Nullable
  @Override
  public abstract Artifact getInterfaceLibrary();

  @Nullable
  @Override
  public abstract Artifact getResolvedSymlinkInterfaceLibrary();

  @Override
  public abstract boolean getAlwayslink();

  // TODO(plf): This is just needed for Go, do not expose to Starlark and try to remove it. This was
  // introduced to let a linker input declare that it needs debug info in the executable.
  // Specifically, this was introduced for linking Go into a C++ binary when using the gccgo
  // compiler.
  abstract boolean getMustKeepDebug();

  abstract boolean getDisableWholeArchive();

  @Override
  public void debugPrint(Printer printer) {
    printer.append("<LibraryToLink(");
    printer.append(
        Joiner.on(", ")
            .skipNulls()
            .join(
                mapEntry("object", getObjectFiles()),
                mapEntry("pic_objects", getPicObjectFiles()),
                mapEntry("static_library", getStaticLibrary()),
                mapEntry("pic_static_library", getPicStaticLibrary()),
                mapEntry("dynamic_library", getDynamicLibrary()),
                mapEntry("resolved_symlink_dynamic_library", getResolvedSymlinkDynamicLibrary()),
                mapEntry("interface_library", getInterfaceLibrary()),
                mapEntry(
                    "resolved_symlink_interface_library", getResolvedSymlinkInterfaceLibrary()),
                mapEntry("alwayslink", getAlwayslink())));
    printer.append(")>");
  }

  private static String mapEntry(String keyName, Object value) {
    if (value == null) {
      return null;
    } else {
      return keyName + "=" + value;
    }
  }

  public static Builder builder() {
    return new AutoValue_LibraryToLink.Builder()
        .setMustKeepDebug(false)
        .setAlwayslink(false)
        .setDisableWholeArchive(false);
  }

  LinkerInputs.LibraryToLink getStaticLibraryToLink() {
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
            getMustKeepDebug(),
            getDisableWholeArchive());
    return staticLibraryToLink;
  }

  LinkerInputs.LibraryToLink getPicStaticLibraryToLink() {
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
            getMustKeepDebug(),
            getDisableWholeArchive());
    return picStaticLibraryToLink;
  }

  LinkerInputs.LibraryToLink getDynamicLibraryToLink() {
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
              getMustKeepDebug(),
              getDisableWholeArchive());
    }
    return dynamicLibraryToLink;
  }

  LinkerInputs.LibraryToLink getInterfaceLibraryToLink() {
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
              getMustKeepDebug(),
              getDisableWholeArchive());
    }
    return interfaceLibraryToLink;
  }

  public static List<Artifact> getDynamicLibrariesForRuntime(
      boolean linkingStatically, Iterable<LibraryToLink> libraries) {
    ImmutableList.Builder<Artifact> dynamicLibrariesForRuntimeBuilder = ImmutableList.builder();
    for (LibraryToLink libraryToLink : libraries) {
      Artifact artifact = libraryToLink.getDynamicLibraryForRuntimeOrNull(linkingStatically);
      if (artifact != null) {
        dynamicLibrariesForRuntimeBuilder.add(artifact);
      }
    }
    return dynamicLibrariesForRuntimeBuilder.build();
  }

  public static List<Artifact> getDynamicLibrariesForLinking(NestedSet<LibraryToLink> libraries) {
    ImmutableList.Builder<Artifact> dynamicLibrariesForLinkingBuilder = ImmutableList.builder();
    for (LibraryToLink libraryToLink : libraries.toList()) {
      if (libraryToLink.getInterfaceLibrary() != null) {
        dynamicLibrariesForLinkingBuilder.add(libraryToLink.getInterfaceLibrary());
      } else if (libraryToLink.getDynamicLibrary() != null) {
        dynamicLibrariesForLinkingBuilder.add(libraryToLink.getDynamicLibrary());
      }
    }
    return dynamicLibrariesForLinkingBuilder.build();
  }

  public abstract Builder toBuilder();

  /** Builder for LibraryToLink. */
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

    public abstract Builder setResolvedSymlinkDynamicLibrary(
        Artifact resolvedSymlinkDynamicLibrary);

    public abstract Builder setInterfaceLibrary(Artifact interfaceLibrary);

    public abstract Builder setResolvedSymlinkInterfaceLibrary(
        Artifact resolvedSymlinkInterfaceLibrary);

    public abstract Builder setAlwayslink(boolean alwayslink);

    public abstract Builder setMustKeepDebug(boolean mustKeepDebug);

    public abstract Builder setDisableWholeArchive(boolean disableWholeArchive);

    // Methods just for validation, not to be called externally.
    abstract LibraryToLink autoBuild();

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

    public LibraryToLink build() {
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
