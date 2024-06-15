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

package com.google.devtools.build.lib.rules.cpp;

import com.google.auto.value.AutoValue;
import com.google.auto.value.extension.memoized.Memoized;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.cpp.LegacyLinkerInputs.LibraryInput;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.LibraryToLinkApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;

/** Encapsulates information for linking a library. */
// The AutoValue implementation of this class already has a sizeable number of fields, meaning that
// instances have a surprising memory cost.
@Immutable
public abstract class LibraryToLink implements LibraryToLinkApi<Artifact, LtoBackendArtifacts> {

  public static ImmutableList<Artifact> getDynamicLibrariesForRuntime(
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

  public static ImmutableList<Artifact> getDynamicLibrariesForLinking(
      NestedSet<LibraryToLink> libraries) {
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

  private LibraryToLink() {}

  public abstract String getLibraryIdentifier();

  @StarlarkMethod(name = "library_identifier", documented = false, useStarlarkThread = true)
  public String getLibraryIdentifierForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return getLibraryIdentifier();
  }

  @Nullable
  public abstract ImmutableList<Artifact> getObjectFiles();

  @Nullable
  public abstract ImmutableMap<Artifact, LtoBackendArtifacts> getSharedNonLtoBackends();

  @Nullable
  public abstract LtoCompilationContext getLtoCompilationContext();

  @StarlarkMethod(
      name = "lto_compilation_context",
      documented = false,
      useStarlarkThread = true,
      allowReturnNones = true)
  @Nullable
  public LtoCompilationContext getLtoCompilationContextForStarlark(StarlarkThread thread)
      throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return getLtoCompilationContext();
  }

  @Nullable
  public abstract ImmutableList<Artifact> getPicObjectFiles();

  @Nullable
  public abstract ImmutableMap<Artifact, LtoBackendArtifacts> getPicSharedNonLtoBackends();

  @Nullable
  public abstract LtoCompilationContext getPicLtoCompilationContext();

  @StarlarkMethod(
      name = "pic_lto_compilation_context",
      documented = false,
      useStarlarkThread = true,
      allowReturnNones = true)
  @Nullable
  public LtoCompilationContext getPicLtoCompilationContextForStarlark(StarlarkThread thread)
      throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return getPicLtoCompilationContext();
  }

  public abstract AutoLibraryToLink.Builder toBuilder();

  @Override
  public final boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  @Nullable
  public final Artifact getDynamicLibraryForRuntimeOrNull(boolean linkingStatically) {
    if (getDynamicLibrary() == null) {
      return null;
    }
    if (linkingStatically && (getStaticLibrary() != null || getPicStaticLibrary() != null)) {
      return null;
    }
    return getDynamicLibrary();
  }

  @Override
  public final Sequence<Artifact> getObjectFilesForStarlark() {
    ImmutableList<Artifact> objectFiles = getObjectFiles();
    return objectFiles == null ? StarlarkList.empty() : StarlarkList.immutableCopyOf(objectFiles);
  }

  @StarlarkMethod(
      name = "objects_private",
      allowReturnNones = true,
      documented = false,
      useStarlarkThread = true)
  @Nullable
  public final Sequence<Artifact> getObjectFilesForStarlarkPrivate(StarlarkThread thread)
      throws EvalException {
    // Returning None here is essential for start-end library functionality. Object files are set
    // to None when calling cc_common.create_library_to_link with empty object files. This signifies
    // to start-end that an archive needs to be used.
    // On the other hand cc_common.link will set object files to exactly what's in the archive.
    // Start-end library functionality may correctly expand the object files. In case they are
    // empty,
    // this means also the archive is empty.
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    ImmutableList<Artifact> objectFiles = getObjectFiles();
    return objectFiles == null ? null : StarlarkList.immutableCopyOf(objectFiles);
  }

  @Override
  public final Sequence<Artifact> getLtoBitcodeFilesForStarlark() {
    LtoCompilationContext ctx = getLtoCompilationContext();
    return ctx == null ? StarlarkList.empty() : StarlarkList.immutableCopyOf(ctx.getBitcodeFiles());
  }

  @StarlarkMethod(
      name = "shared_non_lto_backends",
      documented = false,
      allowReturnNones = true,
      useStarlarkThread = true)
  @Nullable
  public final Dict<Artifact, LtoBackendArtifacts> getSharedNonLtoBackendsForStarlark(
      StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    ImmutableMap<Artifact, LtoBackendArtifacts> backends = getSharedNonLtoBackends();
    return backends != null ? Dict.immutableCopyOf(backends) : null;
  }

  @Override
  public final Sequence<Artifact> getPicObjectFilesForStarlark() {
    ImmutableList<Artifact> objectFiles = getPicObjectFiles();
    return objectFiles == null ? StarlarkList.empty() : StarlarkList.immutableCopyOf(objectFiles);
  }

  @StarlarkMethod(
      name = "pic_objects_private",
      allowReturnNones = true,
      documented = false,
      useStarlarkThread = true)
  @Nullable
  public final Sequence<Artifact> getPicObjectFilesForStarlarkPrivate(StarlarkThread thread)
      throws EvalException {
    // See comment on getObjectFilesForStarlarkPrivate also
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    ImmutableList<Artifact> objectFiles = getPicObjectFiles();
    return objectFiles == null ? null : StarlarkList.immutableCopyOf(objectFiles);
  }

  @Override
  public final Sequence<Artifact> getPicLtoBitcodeFilesForStarlark() {
    LtoCompilationContext ctx = getPicLtoCompilationContext();
    return ctx == null ? StarlarkList.empty() : StarlarkList.immutableCopyOf(ctx.getBitcodeFiles());
  }

  @StarlarkMethod(
      name = "pic_shared_non_lto_backends",
      documented = false,
      allowReturnNones = true,
      useStarlarkThread = true)
  @Nullable
  public final Dict<Artifact, LtoBackendArtifacts> getPicSharedNonLtoBackendsForStarlark(
      StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    ImmutableMap<Artifact, LtoBackendArtifacts> backends = getPicSharedNonLtoBackends();
    return backends != null ? Dict.immutableCopyOf(backends) : null;
  }

  // TODO(b/331164666): This can be removed after cc_common.link is in Starlark
  LibraryInput getStaticLibraryInput() {
    return LegacyLinkerInputs.newInputLibrary(
        Preconditions.checkNotNull(getStaticLibrary(), this),
        getAlwayslink()
            ? ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY
            : ArtifactCategory.STATIC_LIBRARY,
        getLibraryIdentifier(),
        getObjectFiles(),
        getLtoCompilationContext(),
        getSharedNonLtoBackends(),
        getMustKeepDebug(),
        getDisableWholeArchive());
  }

  // TODO(b/331164666): This can be removed after cc_common.link is in Starlark
  LibraryInput getPicStaticLibraryInput() {
    return LegacyLinkerInputs.newInputLibrary(
        Preconditions.checkNotNull(getPicStaticLibrary(), this),
        getAlwayslink()
            ? ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY
            : ArtifactCategory.STATIC_LIBRARY,
        getLibraryIdentifier(),
        getPicObjectFiles(),
        getPicLtoCompilationContext(),
        getPicSharedNonLtoBackends(),
        getMustKeepDebug(),
        getDisableWholeArchive());
  }

  // TODO(b/331164666): This can be removed after cc_common.link is in Starlark
  LibraryInput getDynamicLibraryInput() {
    Artifact dynamicLibrary = Preconditions.checkNotNull(getDynamicLibrary(), this);
    if (getResolvedSymlinkDynamicLibrary() != null) {
      return LegacyLinkerInputs.solibLibraryInput(
          dynamicLibrary, getResolvedSymlinkDynamicLibrary(), getLibraryIdentifier());
    }
    return LegacyLinkerInputs.newInputLibrary(
        dynamicLibrary,
        ArtifactCategory.DYNAMIC_LIBRARY,
        getLibraryIdentifier(),
        /* objectFiles= */ ImmutableSet.of(),
        LtoCompilationContext.EMPTY,
        /* sharedNonLtoBackends= */ ImmutableMap.of(),
        getMustKeepDebug(),
        getDisableWholeArchive());
  }

  // TODO(b/331164666): This can be removed after cc_common.link is in Starlark
  LibraryInput getInterfaceLibraryInput() {
    Artifact interfaceLibrary = Preconditions.checkNotNull(getInterfaceLibrary(), this);
    if (getResolvedSymlinkInterfaceLibrary() != null) {
      return LegacyLinkerInputs.solibLibraryInput(
          interfaceLibrary, getResolvedSymlinkInterfaceLibrary(), getLibraryIdentifier());
    }
    return LegacyLinkerInputs.newInputLibrary(
        interfaceLibrary,
        ArtifactCategory.INTERFACE_LIBRARY,
        getLibraryIdentifier(),
        /* objectFiles= */ ImmutableSet.of(),
        LtoCompilationContext.EMPTY,
        /* sharedNonLtoBackends= */ ImmutableMap.of(),
        getMustKeepDebug(),
        getDisableWholeArchive());
  }

  abstract boolean getMustKeepDebug();

  // TODO(b/338618120): This is just needed for Go, do not expose to Starlark and try to remove it.
  // This was introduced to let a linker input declare that it needs debug info in the executable.
  // Specifically, this was introduced for linking Go into a C++ binary when using the gccgo
  // compiler.
  @StarlarkMethod(name = "must_keep_debug", documented = false, useStarlarkThread = true)
  public final boolean getMustKeepDebugForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return getMustKeepDebug();
  }

  abstract boolean getDisableWholeArchive();

  @StarlarkMethod(name = "disable_whole_archive", documented = false, useStarlarkThread = true)
  public boolean getDisableWholeArchiveForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return getDisableWholeArchive();
  }

  @Override
  public final void debugPrint(Printer printer, StarlarkThread thread) {
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

  @Nullable
  private static String mapEntry(String keyName, @Nullable Object value) {
    return value == null ? null : keyName + "=" + value;
  }

  public static AutoLibraryToLink.Builder builder() {
    return new AutoValue_LibraryToLink_AutoLibraryToLink.Builder()
        .setMustKeepDebug(false)
        .setAlwayslink(false)
        .setDisableWholeArchive(false);
  }

  /** Builder for {@link LibraryToLink}. */
  public interface Builder {

    AutoLibraryToLink.Builder setLibraryIdentifier(String libraryIdentifier);

    AutoLibraryToLink.Builder setStaticLibrary(Artifact staticLibrary);

    AutoLibraryToLink.Builder setObjectFiles(ImmutableList<Artifact> objectFiles);

    AutoLibraryToLink.Builder setLtoCompilationContext(LtoCompilationContext ltoCompilationContext);

    AutoLibraryToLink.Builder setSharedNonLtoBackends(
        ImmutableMap<Artifact, LtoBackendArtifacts> sharedNonLtoBackends);

    AutoLibraryToLink.Builder setPicStaticLibrary(Artifact picStaticLibrary);

    AutoLibraryToLink.Builder setPicObjectFiles(ImmutableList<Artifact> picObjectFiles);

    AutoLibraryToLink.Builder setPicLtoCompilationContext(
        LtoCompilationContext picLtoCompilationContext);

    AutoLibraryToLink.Builder setPicSharedNonLtoBackends(
        ImmutableMap<Artifact, LtoBackendArtifacts> picSharedNonLtoBackends);

    AutoLibraryToLink.Builder setDynamicLibrary(Artifact dynamicLibrary);

    AutoLibraryToLink.Builder setResolvedSymlinkDynamicLibrary(
        Artifact resolvedSymlinkDynamicLibrary);

    AutoLibraryToLink.Builder setInterfaceLibrary(Artifact interfaceLibrary);

    AutoLibraryToLink.Builder setResolvedSymlinkInterfaceLibrary(
        Artifact resolvedSymlinkInterfaceLibrary);

    AutoLibraryToLink.Builder setAlwayslink(boolean alwayslink);

    AutoLibraryToLink.Builder setMustKeepDebug(boolean mustKeepDebug);

    AutoLibraryToLink.Builder setDisableWholeArchive(boolean disableWholeArchive);

    LibraryToLink build();
  }

  /** {@link AutoValue}-backed implementation. */
  @AutoValue
  abstract static class AutoLibraryToLink extends LibraryToLink {

    @Nullable
    @Override // Remove @StarlarkMethod.
    public abstract Artifact getStaticLibrary();

    @Nullable
    @Override // Remove @StarlarkMethod.
    public abstract Artifact getPicStaticLibrary();

    @Nullable
    @Override // Remove @StarlarkMethod.
    public abstract Artifact getDynamicLibrary();

    @Nullable
    @Override // Remove @StarlarkMethod.
    public abstract Artifact getResolvedSymlinkDynamicLibrary();

    @Nullable
    @Override // Remove @StarlarkMethod.
    public abstract Artifact getInterfaceLibrary();

    @Nullable
    @Override // Remove @StarlarkMethod.
    public abstract Artifact getResolvedSymlinkInterfaceLibrary();

    @Override // Remove @StarlarkMethod.
    public abstract boolean getAlwayslink();

    @Memoized
    @Override
    LibraryInput getStaticLibraryInput() {
      return super.getStaticLibraryInput();
    }

    @Memoized
    @Override
    LibraryInput getPicStaticLibraryInput() {
      return super.getPicStaticLibraryInput();
    }

    @Memoized
    @Override
    LibraryInput getDynamicLibraryInput() {
      return super.getDynamicLibraryInput();
    }

    @Memoized
    @Override
    LibraryInput getInterfaceLibraryInput() {
      return super.getInterfaceLibraryInput();
    }

    @AutoValue.Builder
    public abstract static class Builder implements LibraryToLink.Builder {

      Builder() {}

      abstract AutoLibraryToLink autoBuild();

      @Override
      public final LibraryToLink build() {
        LibraryToLink result = autoBuild();
        Preconditions.checkNotNull(result.getLibraryIdentifier(), result);
        Preconditions.checkState(
            result.getResolvedSymlinkDynamicLibrary() == null || result.getDynamicLibrary() != null,
            result);
        Preconditions.checkState(
            result.getResolvedSymlinkInterfaceLibrary() == null
                || result.getResolvedSymlinkInterfaceLibrary() != null,
            result);
        Preconditions.checkState(
            result.getStaticLibrary() != null
                || result.getPicStaticLibrary() != null
                || result.getDynamicLibrary() != null
                || result.getInterfaceLibrary() != null,
            result);

        return result;
      }
    }
  }
}
