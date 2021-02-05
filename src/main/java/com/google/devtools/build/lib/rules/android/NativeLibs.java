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

import com.google.common.base.Preconditions;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Multimap;
import com.google.common.collect.MultimapBuilder;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppHelper;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
import com.google.devtools.build.lib.rules.nativedeps.NativeDepsHelper;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
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
  public static final NativeLibs EMPTY = new NativeLibs(ImmutableMap.of(), null);

  private static String getLibDirName(ConfiguredTargetAndData dep) {
    BuildConfiguration configuration = dep.getConfiguration();
    String name = configuration.getFragment(AndroidConfiguration.class).getCpu();
    if (configuration.getFragment(AndroidConfiguration.class).isHwasan()) {
      name += "-hwasan";
    }
    return name;
  }

  public static NativeLibs fromLinkedNativeDeps(
      RuleContext ruleContext,
      ImmutableList<String> depsAttributes,
      String nativeDepsFileName,
      CppSemantics cppSemantics)
      throws InterruptedException, RuleErrorException {
    Map<String, ConfiguredTargetAndData> toolchainsByLibDir = new LinkedHashMap<>();
    for (ConfiguredTargetAndData toolchainDep :
        ruleContext.getConfiguredTargetAndDataMap().get("$cc_toolchain_split")) {
      toolchainsByLibDir.put(getLibDirName(toolchainDep), toolchainDep);
    }

    // treeKeys() means that the resulting map sorts the entries by key, which is necessary to
    // ensure determinism.
    Multimap<String, ConfiguredTargetAndData> depsByLibDir =
        MultimapBuilder.treeKeys().arrayListValues().build();
    for (String depsAttr : depsAttributes) {
      for (ConfiguredTargetAndData dep :
          ruleContext.getConfiguredTargetAndDataMap().get(depsAttr)) {
        depsByLibDir.put(getLibDirName(dep), dep);
      }
    }

    String nativeDepsLibraryBasename = null;
    Map<String, NestedSet<Artifact>> result = new LinkedHashMap<>();
    for (Map.Entry<String, Collection<ConfiguredTargetAndData>> entry :
        depsByLibDir.asMap().entrySet()) {
      ConfiguredTargetAndData toolchainDep = toolchainsByLibDir.get(entry.getKey());
      CcToolchainProvider toolchain =
          CppHelper.getToolchain(ruleContext, toolchainDep.getConfiguredTarget());

      CcInfo ccInfo =
          AndroidCommon.getCcInfo(
              Collections2.transform(
                  entry.getValue(), ConfiguredTargetAndData::getConfiguredTarget),
              ImmutableList.of("-Wl,-soname=lib" + ruleContext.getLabel().getName()),
              ruleContext.getLabel(),
              ruleContext.getSymbolGenerator());

      Artifact nativeDepsLibrary =
          NativeDepsHelper.linkAndroidNativeDepsIfPresent(
              ruleContext, ccInfo, toolchainDep.getConfiguration(), toolchain, cppSemantics);

      NestedSetBuilder<Artifact> librariesBuilder = NestedSetBuilder.stableOrder();
      if (nativeDepsLibrary != null) {
        librariesBuilder.add(nativeDepsLibrary);
        nativeDepsLibraryBasename = nativeDepsLibrary.getExecPath().getBaseName();
      }
      librariesBuilder.addAll(
          filterUniqueSharedLibraries(
              ruleContext, nativeDepsLibrary, ccInfo.getCcLinkingContext().getLibraries()));
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
      Artifact nativeDepsName =
          ruleContext.getUniqueDirectoryArtifact(
              "nativedeps_filename", nativeDepsFileName, ruleContext.getBinOrGenfilesDirectory());
      ruleContext.registerAction(
          FileWriteAction.create(ruleContext, nativeDepsName, nativeDepsLibraryBasename, false));

      return new NativeLibs(ImmutableMap.copyOf(result), nativeDepsName);
    }
  }

  // Map from architecture (CPU folder to place the library in) to libraries for that CPU
  private final ImmutableMap<String, NestedSet<Artifact>> nativeLibs;
  @Nullable private final Artifact nativeLibsName;

  private NativeLibs(
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
      RuleContext ruleContext, Artifact linkedLibrary, NestedSet<LibraryToLink> libraries) {
    Map<String, Artifact> basenames = new HashMap<>();
    Set<Artifact> artifacts = new HashSet<>();
    if (linkedLibrary != null) {
      basenames.put(linkedLibrary.getExecPath().getBaseName(), linkedLibrary);
    }
    for (LibraryToLink linkerInput : libraries.toList()) {
      if (linkerInput.getPicStaticLibrary() != null || linkerInput.getStaticLibrary() != null) {
        // This is not a shared library and will not be loaded by Android, so skip it.
        continue;
      }
      Artifact artifact = null;
      if (linkerInput.getInterfaceLibrary() != null) {
        if (linkerInput.getResolvedSymlinkInterfaceLibrary() != null) {
          artifact = linkerInput.getResolvedSymlinkInterfaceLibrary();
        } else {
          artifact = linkerInput.getInterfaceLibrary();
        }
      } else {
        if (linkerInput.getResolvedSymlinkDynamicLibrary() != null) {
          artifact = linkerInput.getResolvedSymlinkDynamicLibrary();
        } else {
          artifact = linkerInput.getDynamicLibrary();
        }
      }
      Preconditions.checkNotNull(artifact);

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
