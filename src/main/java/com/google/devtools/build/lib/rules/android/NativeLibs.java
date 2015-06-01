// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.cpp.CcLinkParams;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.LinkerInput;
import com.google.devtools.build.lib.rules.nativedeps.NativeDepsHelper;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/** Represents the collection of native libraries (.so) to be installed in the APK. */
public final class NativeLibs {
  public static final NativeLibs EMPTY =
      new NativeLibs(ImmutableMap.<String, Iterable<Artifact>>of(), null);

  public static NativeLibs fromPrecompiledObjects(
      RuleContext ruleContext, Multimap<String, TransitiveInfoCollection> depsByArchitecture) {
    ImmutableMap.Builder<String, Iterable<Artifact>> builder = ImmutableMap.builder();
    for (Map.Entry<String, Collection<TransitiveInfoCollection>> entry :
        depsByArchitecture.asMap().entrySet()) {
      NestedSet<LinkerInput> nativeLibraries =
          AndroidCommon.collectTransitiveNativeLibraries(entry.getValue());
      builder.put(entry.getKey(), checkUniqueBaseNames(ruleContext, nativeLibraries));
    }
    return new NativeLibs(builder.build(), null);
  }

  public static NativeLibs fromLinkedNativeDeps(
      RuleContext ruleContext,
      String nativeDepsFileName,
      Multimap<String, TransitiveInfoCollection> depsByArchitecture,
      Map<String, CcToolchainProvider> toolchainMap,
      Map<String, BuildConfiguration> configurationMap) {
    Map<String, Iterable<Artifact>> result = new LinkedHashMap<>();
    for (Map.Entry<String, Collection<TransitiveInfoCollection>> entry :
        depsByArchitecture.asMap().entrySet()) {
      CcLinkParams linkParams = AndroidCommon.getCcLinkParamsStore(entry.getValue())
          .get(/* linkingStatically */ true, /* linkShared */ true);
      Artifact nativeDepsLibrary = NativeDepsHelper.maybeCreateAndroidNativeDepsAction(
          ruleContext, linkParams, configurationMap.get(entry.getKey()),
          toolchainMap.get(entry.getKey()));
      if (nativeDepsLibrary != null) {
        result.put(entry.getKey(), ImmutableList.of(nativeDepsLibrary));
      }
    }
    if (result.isEmpty()) {
      return NativeLibs.EMPTY;
    } else {
      Artifact anyNativeLibrary =
          result.entrySet().iterator().next().getValue().iterator().next();
      // The native deps name file must be the only file in its directory because ApkBuilder does
      // not have an option to add a particular file to the .apk, only one to add every file in a
      // particular directory.
      Artifact nativeDepsName = ruleContext.getAnalysisEnvironment().getDerivedArtifact(
          ruleContext.getUniqueDirectory("nativedeps_filename").getRelative(nativeDepsFileName),
          ruleContext.getBinOrGenfilesDirectory());
      ruleContext.registerAction(new FileWriteAction(ruleContext.getActionOwner(), nativeDepsName,
          anyNativeLibrary.getExecPath().getBaseName(), false));

      return new NativeLibs(ImmutableMap.copyOf(result), nativeDepsName);
    }
  }

  // Map from architecture (CPU folder to place the library in) to libraries for that CPU
  private final Map<String, Iterable<Artifact>> nativeLibs;
  private final Artifact nativeLibsName;

  private NativeLibs(Map<String, Iterable<Artifact>> nativeLibs, Artifact nativeLibsName) {
    this.nativeLibs = nativeLibs;
    this.nativeLibsName = nativeLibsName;
  }

  /**
   * Returns a map from the name of the architecture (CPU folder to place the library in) to the
   * set of libraries for that architecture.
   */
  public Map<String, Iterable<Artifact>> getMap() {
    return nativeLibs;
  }

  public ImmutableList<Artifact> createApkBuilderSymlinks(RuleContext ruleContext) {
      ImmutableList.Builder<Artifact> result = ImmutableList.builder();
    for (Map.Entry<String, Iterable<Artifact>> entry : nativeLibs.entrySet()) {
      String arch = entry.getKey();
      for (Artifact lib : entry.getValue()) {
        Artifact symlink = AndroidBinary.getDxArtifact(ruleContext,
            "native_symlinks/" + arch + "/" + lib.getExecPath().getBaseName());
        ruleContext.registerAction(new SymlinkAction(
            ruleContext.getActionOwner(), lib, symlink,
            "Symlinking Android native library for " + ruleContext.getLabel()));
        result.add(symlink);
      }
    }

    return result.build();
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

  private static Iterable<Artifact> checkUniqueBaseNames(
      RuleContext ruleContext, NestedSet<LinkerInput> libraries) {
    Map<String, Artifact> basenames = new HashMap<>();
    Set<Artifact> artifacts = new HashSet<>();
    for (LinkerInput linkerInput : libraries) {
      Artifact artifact = linkerInput.getOriginalLibraryArtifact();
      if (!artifacts.add(artifact)) {
        // We have already reached this library, e.g., through a different solib symlink.
        continue;
      }
      String basename = artifact.getExecPath().getBaseName();
      Artifact oldArtifact = basenames.put(basename, artifact);
      if (oldArtifact != null) {
        // Without linking, there may be name collisions in the libraries which were provided, so
        // check for this at this step.
        ruleContext.ruleError("Each library in the transitive closure must have a unique "
            + "basename, but two libraries had the basename '" + basename + "': "
            + artifact.prettyPrint() + " and " + oldArtifact.prettyPrint());
      }
    }
    return artifacts;
  }
}
