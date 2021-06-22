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

import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.google.auto.value.AutoValue;
import com.google.auto.value.extension.memoized.Memoized;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.LibraryToLinkApi;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;

/** Encapsulates information for linking a library. */
// The AutoValue implementation of this class already has a sizeable number of fields, meaning that
// instances have a surprising memory cost. We may benefit from having more specialized
// implementations similar to StaticOnlyLibraryToLink, for cases when certain fields are always
// null. Consider this before adding additional fields to this class. See b/181991741.
@Immutable
public abstract class LibraryToLink implements LibraryToLinkApi<Artifact, LtoBackendArtifacts> {

  public static final Depset.ElementType TYPE = Depset.ElementType.of(LibraryToLink.class);

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

  @Nullable
  public abstract ImmutableList<Artifact> getObjectFiles();

  @Nullable
  public abstract ImmutableMap<Artifact, LtoBackendArtifacts> getSharedNonLtoBackends();

  @Nullable
  public abstract LtoCompilationContext getLtoCompilationContext();

  @Nullable
  public abstract ImmutableList<Artifact> getPicObjectFiles();

  @Nullable
  public abstract ImmutableMap<Artifact, LtoBackendArtifacts> getPicSharedNonLtoBackends();

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

  @Override
  public final Sequence<Artifact> getLtoBitcodeFilesForStarlark() {
    LtoCompilationContext ctx = getLtoCompilationContext();
    return ctx == null ? StarlarkList.empty() : StarlarkList.immutableCopyOf(ctx.getBitcodeFiles());
  }

  @Override
  public final boolean getMustKeepDebugForStarlark(StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return getMustKeepDebug();
  }

  @Override
  public final Dict<Artifact, LtoBackendArtifacts> getSharedNonLtoBackendsForStarlark(
      StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return Dict.immutableCopyOf(getSharedNonLtoBackends());
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

  @Override
  public final Dict<Artifact, LtoBackendArtifacts> getPicSharedNonLtoBackendsForStarlark(
      StarlarkThread thread) throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return Dict.immutableCopyOf(getPicSharedNonLtoBackends());
  }

  LinkerInputs.LibraryToLink getStaticLibraryToLink() {
    return LinkerInputs.newInputLibrary(
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

  LinkerInputs.LibraryToLink getPicStaticLibraryToLink() {
    return LinkerInputs.newInputLibrary(
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

  LinkerInputs.LibraryToLink getDynamicLibraryToLink() {
    Artifact dynamicLibrary = Preconditions.checkNotNull(getDynamicLibrary(), this);
    if (getResolvedSymlinkDynamicLibrary() != null) {
      return LinkerInputs.solibLibraryToLink(
          dynamicLibrary, getResolvedSymlinkDynamicLibrary(), getLibraryIdentifier());
    }
    return LinkerInputs.newInputLibrary(
        dynamicLibrary,
        ArtifactCategory.DYNAMIC_LIBRARY,
        getLibraryIdentifier(),
        /*objectFiles=*/ ImmutableSet.of(),
        LtoCompilationContext.EMPTY,
        /*sharedNonLtoBackends=*/ ImmutableMap.of(),
        getMustKeepDebug(),
        getDisableWholeArchive());
  }

  LinkerInputs.LibraryToLink getInterfaceLibraryToLink() {
    Artifact interfaceLibrary = Preconditions.checkNotNull(getInterfaceLibrary(), this);
    if (getResolvedSymlinkInterfaceLibrary() != null) {
      return LinkerInputs.solibLibraryToLink(
          interfaceLibrary, getResolvedSymlinkInterfaceLibrary(), getLibraryIdentifier());
    }
    return LinkerInputs.newInputLibrary(
        interfaceLibrary,
        ArtifactCategory.INTERFACE_LIBRARY,
        getLibraryIdentifier(),
        /*objectFiles=*/ ImmutableSet.of(),
        LtoCompilationContext.EMPTY,
        /*sharedNonLtoBackends=*/ ImmutableMap.of(),
        getMustKeepDebug(),
        getDisableWholeArchive());
  }

  // TODO(plf): This is just needed for Go, do not expose to Starlark and try to remove it. This was
  // introduced to let a linker input declare that it needs debug info in the executable.
  // Specifically, this was introduced for linking Go into a C++ binary when using the gccgo
  // compiler.
  abstract boolean getMustKeepDebug();

  abstract boolean getDisableWholeArchive();

  @Override
  public final void debugPrint(Printer printer) {
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

  private static String mapEntry(String keyName, @Nullable Object value) {
    return value == null ? null : keyName + "=" + value;
  }

  public static AutoLibraryToLink.Builder builder() {
    return new AutoValue_LibraryToLink_AutoLibraryToLink.Builder()
        .setMustKeepDebug(false)
        .setAlwayslink(false)
        .setDisableWholeArchive(false);
  }

  /**
   * Creates a {@link LibraryToLink} that has only {@link #getStaticLibrary} and no other optional
   * fields.
   */
  public static LibraryToLink staticOnly(Artifact staticLibrary) {
    return StaticOnlyLibraryToLink.cache.get(staticLibrary);
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
    LinkerInputs.LibraryToLink getStaticLibraryToLink() {
      return super.getStaticLibraryToLink();
    }

    @Memoized
    @Override
    LinkerInputs.LibraryToLink getPicStaticLibraryToLink() {
      return super.getPicStaticLibraryToLink();
    }

    @Memoized
    @Override
    LinkerInputs.LibraryToLink getDynamicLibraryToLink() {
      return super.getDynamicLibraryToLink();
    }

    @Memoized
    @Override
    LinkerInputs.LibraryToLink getInterfaceLibraryToLink() {
      return super.getInterfaceLibraryToLink();
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
            (result.getObjectFiles() == null
                    && result.getLtoCompilationContext() == null
                    && result.getSharedNonLtoBackends() == null)
                || result.getStaticLibrary() != null,
            result);
        Preconditions.checkState(
            (result.getPicObjectFiles() == null
                    && result.getPicLtoCompilationContext() == null
                    && result.getPicSharedNonLtoBackends() == null)
                || result.getPicStaticLibrary() != null,
            result);
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

        // Static-only instances must always return StaticOnlyLibraryToLink to preserve equality.
        if (result.getStaticLibrary() != null
            && !result.getAlwayslink()
            && !result.getMustKeepDebug()
            && !result.getDisableWholeArchive()
            && result.getPicStaticLibrary() == null
            && result.getDynamicLibrary() == null
            && result.getInterfaceLibrary() == null
            && result.getSharedNonLtoBackends() == null
            && result.getPicObjectFiles() == null
            && result.getPicLtoCompilationContext() == null) {
          Artifact staticLibrary = result.getStaticLibrary();
          String libraryIdentifier = result.getLibraryIdentifier();

          // Try to reuse an existing instance if possible.
          StaticOnlyLibraryToLink existing =
              StaticOnlyLibraryToLink.cache.getIfPresent(staticLibrary);
          if (existing != null && existing.getLibraryIdentifier().equals(libraryIdentifier)) {
            return existing;
          }

          return new AutoValue_LibraryToLink_StaticOnlyLibraryToLink(
              result.getLibraryIdentifier(), result.getStaticLibrary());
        }

        return result;
      }
    }
  }

  /**
   * Specialized implementation for the case when only {@link #getStaticLibrary} is needed, to save
   * memory compared to {@link AutoLibraryToLink}.
   */
  @AutoValue
  abstract static class StaticOnlyLibraryToLink extends LibraryToLink {

    // Essentially an interner, but keyed on Artifact to defer creating the string identifier.
    private static final LoadingCache<Artifact, StaticOnlyLibraryToLink> cache =
        Caffeine.newBuilder()
            .initialCapacity(BlazeInterners.concurrencyLevel())
            // Needs to use weak keys for identity equality of the artifact. The artifact may not
            // yet have its generating action key set, but Artifact#equals treats unset and set as
            // equal. Reusing an artifact from a previous build is not safe - the generating
            // action key's index may be stale (b/184948206).
            .weakKeys()
            .weakValues()
            .build(
                artifact ->
                    new AutoValue_LibraryToLink_StaticOnlyLibraryToLink(
                        CcLinkingOutputs.libraryIdentifierOf(artifact), artifact));

    @Override // Remove @Nullable.
    public abstract Artifact getStaticLibrary();

    @Nullable
    @Override
    public ImmutableList<Artifact> getObjectFiles() {
      return null;
    }

    @Nullable
    @Override
    public ImmutableMap<Artifact, LtoBackendArtifacts> getSharedNonLtoBackends() {
      return null;
    }

    @Nullable
    @Override
    public LtoCompilationContext getLtoCompilationContext() {
      return null;
    }

    @Nullable
    @Override
    public ImmutableList<Artifact> getPicObjectFiles() {
      return null;
    }

    @Nullable
    @Override
    public ImmutableMap<Artifact, LtoBackendArtifacts> getPicSharedNonLtoBackends() {
      return null;
    }

    @Nullable
    @Override
    public LtoCompilationContext getPicLtoCompilationContext() {
      return null;
    }

    @Nullable
    @Override
    public Artifact getPicStaticLibrary() {
      return null;
    }

    @Nullable
    @Override
    public Artifact getDynamicLibrary() {
      return null;
    }

    @Nullable
    @Override
    public Artifact getResolvedSymlinkDynamicLibrary() {
      return null;
    }

    @Nullable
    @Override
    public Artifact getInterfaceLibrary() {
      return null;
    }

    @Nullable
    @Override
    public Artifact getResolvedSymlinkInterfaceLibrary() {
      return null;
    }

    @Override
    public boolean getAlwayslink() {
      return false;
    }

    @Override
    public AutoLibraryToLink.Builder toBuilder() {
      return builder()
          .setStaticLibrary(getStaticLibrary())
          .setLibraryIdentifier(getLibraryIdentifier());
    }

    @Override
    boolean getMustKeepDebug() {
      return false;
    }

    @Override
    boolean getDisableWholeArchive() {
      return false;
    }
  }
}
