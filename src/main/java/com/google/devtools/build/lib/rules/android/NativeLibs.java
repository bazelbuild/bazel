// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.SourceManifestAction;
import com.google.devtools.build.lib.analysis.SourceManifestAction.ManifestType;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkTreeAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.cpp.CcLinkParams;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppFileTypes;
import com.google.devtools.build.lib.rules.cpp.LinkerInput;
import com.google.devtools.build.lib.rules.nativedeps.NativeDepsHelper;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/** Represents the collection of native libraries (.so) to be installed in the APK. */
@Immutable
public final class NativeLibs {
  public static final NativeLibs EMPTY =
      new NativeLibs(ImmutableMap.<String, NestedSet<Artifact>>of(), null);

  public static NativeLibs fromLinkedNativeDeps(
      RuleContext ruleContext,
      String nativeDepsFileName,
      Multimap<String, TransitiveInfoCollection> depsByArchitecture,
      Map<String, CcToolchainProvider> toolchainMap,
      Map<String, BuildConfiguration> configurationMap)
      throws InterruptedException {
    Map<String, NestedSet<Artifact>> result = new LinkedHashMap<>();
    String nativeDepsLibraryBasename = null;
    for (Map.Entry<String, Collection<TransitiveInfoCollection>> entry :
        depsByArchitecture.asMap().entrySet()) {
      CcLinkParams linkParams =
          AndroidCommon.getCcLinkParamsStore(
                  entry.getValue(),
                  ImmutableList.of("-Wl,-soname=lib" + ruleContext.getLabel().getName()))
              .get(/* linkingStatically */ true, /* linkShared */ true);

      Artifact nativeDepsLibrary =
          NativeDepsHelper.linkAndroidNativeDepsIfPresent(
              ruleContext,
              linkParams,
              configurationMap.get(entry.getKey()),
              toolchainMap.get(entry.getKey()));

      NestedSetBuilder<Artifact> librariesBuilder = NestedSetBuilder.stableOrder();
      if (nativeDepsLibrary != null) {
        librariesBuilder.add(nativeDepsLibrary);
        nativeDepsLibraryBasename = nativeDepsLibrary.getExecPath().getBaseName();
      }
      librariesBuilder.addAll(
          filterUniqueSharedLibraries(ruleContext, nativeDepsLibrary, linkParams.getLibraries()));
      NestedSet<Artifact> libraries = librariesBuilder.build();

      if (!libraries.isEmpty()) {
        result.put(entry.getKey(), libraries);
      }
    }
    if (result.isEmpty()) {
      return NativeLibs.EMPTY;
    } else if (nativeDepsLibraryBasename == null) {
      return new NativeLibs(ImmutableMap.copyOf(result), null);
    } else {
      // The native deps name file must be the only file in its directory because ApkBuilder does
      // not have an option to add a particular file to the .apk, only one to add every file in a
      // particular directory.
      Artifact nativeDepsName = ruleContext.getUniqueDirectoryArtifact(
          "nativedeps_filename", nativeDepsFileName,
          ruleContext.getBinOrGenfilesDirectory());
      ruleContext.registerAction(
          FileWriteAction.create(ruleContext, nativeDepsName, nativeDepsLibraryBasename, false));

      return new NativeLibs(ImmutableMap.copyOf(result), nativeDepsName);
    }
  }

  // Map from architecture (CPU folder to place the library in) to libraries for that CPU
  private final ImmutableMap<String, NestedSet<Artifact>> nativeLibs;
  @Nullable private final Artifact nativeLibsName;

  @VisibleForTesting
  NativeLibs(
      ImmutableMap<String, NestedSet<Artifact>> nativeLibs, @Nullable Artifact nativeLibsName) {
    this.nativeLibs = nativeLibs;
    this.nativeLibsName = nativeLibsName;
  }

  /**
   * Returns a map from the name of the architecture (CPU folder to place the library in) to the
   * nested set of libraries for that architecture.
   */
  public Map<String, NestedSet<Artifact>> getMap() {
    return nativeLibs;
  }

  public ImmutableSet<Artifact> getAllNativeLibs() {
    ImmutableSet.Builder<Artifact> result = ImmutableSet.builder();

    for (Iterable<Artifact> libs : nativeLibs.values()) {
      result.addAll(libs);
    }

    return result.build();
  }

  public Pair<Artifact, Runfiles> createApkBuilderSymlinks(RuleContext ruleContext) {
    Map<PathFragment, Artifact> symlinks = new LinkedHashMap<>();
    for (Map.Entry<String, NestedSet<Artifact>> entry : nativeLibs.entrySet()) {
      String arch = entry.getKey();
      for (Artifact lib : entry.getValue()) {
        symlinks.put(PathFragment.create(arch + "/" + lib.getExecPath().getBaseName()), lib);
      }
    }

    if (symlinks.isEmpty()) {
      return null;
    }

    Artifact inputManifest = AndroidBinary.getDxArtifact(ruleContext, "native_symlinks.manifest");
    SourceManifestAction sourceManifestAction = new SourceManifestAction.Builder(
        ruleContext.getWorkspaceName(), ManifestType.SOURCE_SYMLINKS, ruleContext.getActionOwner(),
        inputManifest, ruleContext.getConfiguration().legacyExternalRunfiles())
            .addRootSymlinks(symlinks)
            .build();
    ruleContext.registerAction(sourceManifestAction);
    Artifact outputManifest = AndroidBinary.getDxArtifact(ruleContext, "native_symlinks/MANIFEST");
    Artifact nativeLibsMiddleman =
        ruleContext.getAnalysisEnvironment().getMiddlemanFactory().createRunfilesMiddleman(
            ruleContext.getActionOwner(), null, symlinks.values(),
            ruleContext.getConfiguration().getMiddlemanDirectory(
                ruleContext.getRule().getRepository()), "android_native_libs");

    ruleContext.registerAction(
        new SymlinkTreeAction(
            ruleContext.getActionOwner(),
            inputManifest,
            nativeLibsMiddleman,
            outputManifest,
            false,
            ruleContext.getConfiguration().getLocalShellEnvironment(),
            ruleContext.getConfiguration().runfilesEnabled()));
    return Pair.of(outputManifest, sourceManifestAction.getGeneratedRunfiles());
  }

  /**
   * Returns the artifact containing the names of the native libraries or null if it does not exist.
   *
   * <p>This artifact will be put in the root directory of the .apk and can be used to load the
   * libraries programmatically without knowing their names.
   */
  @Nullable
  public Artifact getName() {
    return nativeLibsName;
  }

  private static Iterable<Artifact> filterUniqueSharedLibraries(
      RuleContext ruleContext, Artifact linkedLibrary, NestedSet<? extends LinkerInput> libraries) {
    Map<String, Artifact> basenames = new HashMap<>();
    Set<Artifact> artifacts = new HashSet<>();
    if (linkedLibrary != null) {
      basenames.put(linkedLibrary.getExecPath().getBaseName(), linkedLibrary);
    }
    for (LinkerInput linkerInput : libraries) {
      String name = linkerInput.getArtifact().getFilename();
      if (!(CppFileTypes.SHARED_LIBRARY.matches(name)
          || CppFileTypes.VERSIONED_SHARED_LIBRARY.matches(name))) {
        // This is not a shared library and will not be loaded by Android, so skip it.
        continue;
      }
      Artifact artifact = linkerInput.getOriginalLibraryArtifact();
      if (!artifacts.add(artifact)) {
        // We have already reached this library, e.g., through a different solib symlink.
        continue;
      }
      String basename = artifact.getExecPath().getBaseName();
      Artifact oldArtifact = basenames.put(basename, artifact);
      if (oldArtifact != null) {
        // There may be name collisions in the libraries which were provided, so
        // check for this at this step.
        ruleContext.ruleError(
            "Each library in the transitive closure must have a unique basename to avoid "
                + "name collisions when packaged into an apk, but two libraries have the basename '"
                + basename
                + "': "
                + artifact.prettyPrint()
                + " and "
                + oldArtifact.prettyPrint()
                + ((oldArtifact.equals(linkedLibrary))
                    ? " (the library compiled for this target)"
                    : ""));
      }
    }
    return artifacts;
  }
}
