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
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.LibraryToLinkApi;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;

/** Encapsulates information for linking a library. */
// The AutoValue implementation of this class already has a sizeable number of fields, meaning that
// instances have a surprising memory cost.
@Immutable
public abstract class LibraryToLink implements LibraryToLinkApi {

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

  @StarlarkMethod(name = "_library_identifier", documented = false, structField = true)
  public abstract String getLibraryIdentifier();

  @Nullable
  @Override
  public abstract Artifact getStaticLibrary();

  @Nullable
  @Override
  public abstract Artifact getPicStaticLibrary();

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

  @Nullable
  public abstract ImmutableList<Artifact> getObjectFiles();

  @Nullable
  public abstract ImmutableMap<Artifact, LtoBackendArtifacts> getSharedNonLtoBackends();

  @StarlarkMethod(
      name = "_lto_compilation_context",
      documented = false,
      structField = true,
      allowReturnNones = true)
  @Nullable
  public abstract LtoCompilationContext getLtoCompilationContext();

  @Nullable
  public abstract ImmutableList<Artifact> getPicObjectFiles();

  @Nullable
  public abstract ImmutableMap<Artifact, LtoBackendArtifacts> getPicSharedNonLtoBackends();

  @StarlarkMethod(
      name = "_pic_lto_compilation_context",
      documented = false,
      structField = true,
      allowReturnNones = true)
  @Nullable
  public abstract LtoCompilationContext getPicLtoCompilationContext();

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

  /**
   * This is essential for start-end library functionality. _contains_objects is False when calling
   * cc_common.create_library_to_link with empty object files. This signifies to start-end that an
   * archive needs to be used. On the other hand cc_common.link will set object files to exactly
   * what's in the archive. Start-end library functionality may correctly expand the object files.
   * In case they are empty, this means also the archive is empty.
   */
  @StarlarkMethod(name = "_contains_objects", documented = false, structField = true)
  public abstract boolean getContainsObjects();

  @Override
  public final Sequence<Artifact> getLtoBitcodeFilesForStarlark() {
    LtoCompilationContext ctx = getLtoCompilationContext();
    return ctx == null ? StarlarkList.empty() : StarlarkList.immutableCopyOf(ctx.getBitcodeFiles());
  }

  @StarlarkMethod(
      name = "_shared_non_lto_backends",
      documented = false,
      allowReturnNones = true,
      structField = true)
  @Nullable
  public final Dict<Artifact, LtoBackendArtifacts> getSharedNonLtoBackendsForStarlark() {
    ImmutableMap<Artifact, LtoBackendArtifacts> backends = getSharedNonLtoBackends();
    return backends != null ? Dict.immutableCopyOf(backends) : null;
  }

  @Override
  public final Sequence<Artifact> getPicObjectFilesForStarlark() {
    ImmutableList<Artifact> objectFiles = getPicObjectFiles();
    return objectFiles == null ? StarlarkList.empty() : StarlarkList.immutableCopyOf(objectFiles);
  }

  @Override
  public final Sequence<Artifact> getPicLtoBitcodeFilesForStarlark() {
    LtoCompilationContext ctx = getPicLtoCompilationContext();
    return ctx == null ? StarlarkList.empty() : StarlarkList.immutableCopyOf(ctx.getBitcodeFiles());
  }

  @StarlarkMethod(
      name = "_pic_shared_non_lto_backends",
      documented = false,
      allowReturnNones = true,
      structField = true)
  @Nullable
  public final Dict<Artifact, LtoBackendArtifacts> getPicSharedNonLtoBackendsForStarlark() {
    ImmutableMap<Artifact, LtoBackendArtifacts> backends = getPicSharedNonLtoBackends();
    return backends != null ? Dict.immutableCopyOf(backends) : null;
  }

  // TODO(b/338618120): This is just needed for Go, do not expose to Starlark and try to remove it.
  // This was introduced to let a linker input declare that it needs debug info in the executable.
  // Specifically, this was introduced for linking Go into a C++ binary when using the gccgo
  // compiler.
  @StarlarkMethod(name = "_must_keep_debug", documented = false, structField = true)
  public abstract boolean getMustKeepDebug();

  @StarlarkMethod(name = "_disable_whole_archive", documented = false, structField = true)
  public abstract boolean getDisableWholeArchive();

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
        .setDisableWholeArchive(false)
        .setContainsObjects(false);
  }

  /** Builder for {@link LibraryToLink}. */
  public interface Builder {

    AutoLibraryToLink.Builder setLibraryIdentifier(String libraryIdentifier);

    AutoLibraryToLink.Builder setStaticLibrary(Artifact staticLibrary);

    AutoLibraryToLink.Builder setObjectFiles(ImmutableList<Artifact> objectFiles);

    AutoLibraryToLink.Builder setLtoCompilationContext(LtoCompilationContext ltoCompilationContext);

    @CanIgnoreReturnValue
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

    AutoLibraryToLink.Builder setContainsObjects(boolean containsObjects);

    LibraryToLink build();
  }

  /** {@link AutoValue}-backed implementation. */
  @AutoValue
  abstract static class AutoLibraryToLink extends LibraryToLink {
    @Override // Remove @StarlarkMethod.
    public abstract boolean getAlwayslink();

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
