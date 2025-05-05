// Copyright 2016 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.PerLabelOptions;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/** Builder class to construct C++ link action. */
public class CppLinkActionBuilder {
  /**
   * Provides ActionConstructionContext, BuildConfigurationValue and methods for creating
   * intermediate and output artifacts for C++ linking.
   *
   * <p>This is unfortunately necessary, because most of the time, these artifacts are well-behaved
   * ones sitting under a package directory, but nativedeps link actions can be shared. In order to
   * avoid creating every artifact here with {@code getShareableArtifact()}, we abstract the
   * artifact creation away.
   *
   * <p>With shareableArtifacts set to true the implementation can create artifacts anywhere.
   *
   * <p>Necessary when the LTO backend actions of libraries should be shareable, and thus cannot be
   * under the package directory.
   *
   * <p>Necessary because the actions of nativedeps libraries should be shareable, and thus cannot
   * be under the package directory.
   */
  public static class LinkActionConstruction {
    private final boolean shareableArtifacts;
    private final ActionConstructionContext context;
    private final BuildConfigurationValue config;

    public ActionConstructionContext getContext() {
      return context;
    }

    public BuildConfigurationValue getConfig() {
      return config;
    }

    LinkActionConstruction(
        ActionConstructionContext context,
        BuildConfigurationValue config,
        boolean shareableArtifacts) {
      this.context = context;
      this.config = config;
      this.shareableArtifacts = shareableArtifacts;
    }

    public Artifact create(PathFragment rootRelativePath) {
      RepositoryName repositoryName = context.getActionOwner().getLabel().getRepository();
      if (shareableArtifacts) {
        return context.getShareableArtifact(
            rootRelativePath, config.getBinDirectory(repositoryName));

      } else {
        return context.getDerivedArtifact(rootRelativePath, config.getBinDirectory(repositoryName));
      }
    }

    public SpecialArtifact createTreeArtifact(PathFragment rootRelativePath) {
      RepositoryName repositoryName = context.getActionOwner().getLabel().getRepository();
      if (shareableArtifacts) {
        return context
            .getAnalysisEnvironment()
            .getTreeArtifact(rootRelativePath, config.getBinDirectory(repositoryName));
      } else {
        return context.getTreeArtifact(rootRelativePath, config.getBinDirectory(repositoryName));
      }
    }

    public ArtifactRoot getBinDirectory() {
      return config.getBinDirectory(context.getActionOwner().getLabel().getRepository());
    }
  }

  public static LinkActionConstruction newActionConstruction(RuleContext context) {
    return new LinkActionConstruction(context, context.getConfiguration(), false);
  }

  public static LinkActionConstruction newActionConstruction(
      ActionConstructionContext context,
      BuildConfigurationValue config,
      boolean shareableArtifacts) {
    return new LinkActionConstruction(context, config, shareableArtifacts);
  }

  /*
   * Create an LtoBackendArtifacts object, using the appropriate constructor depending on whether
   * the associated ThinLTO link will utilize LTO indexing (therefore unique LTO backend actions),
   * or not (and therefore the library being linked will create a set of shared LTO backends).
   */
  private static LtoBackendArtifacts createLtoArtifact(
      LinkActionConstruction linkActionConstruction,
      FeatureConfiguration featureConfiguration,
      CcToolchainProvider toolchain,
      boolean usePicForLtoBackendActions,
      Artifact bitcodeFile,
      @Nullable BitcodeFiles allBitcode,
      PathFragment ltoOutputRootPrefix,
      PathFragment ltoObjRootPrefix,
      boolean createSharedNonLto,
      List<String> argv)
      throws EvalException {
    // Depending on whether LTO indexing is allowed, generate an LTO backend
    // that will be fed the results of the indexing step, or a dummy LTO backend
    // that simply compiles the bitcode into native code without any index-based
    // cross module optimization.
    LinkActionConstruction localLinkActionConstruction = linkActionConstruction;
    if (createSharedNonLto) {
      localLinkActionConstruction =
          new LinkActionConstruction(
              linkActionConstruction.getContext(),
              linkActionConstruction.getConfig(),
              /* shareableArtifacts= */ true);
    }
    BitcodeFiles bitcodeFiles = createSharedNonLto ? null : allBitcode;
    return new LtoBackendArtifacts(
        ltoOutputRootPrefix,
        ltoObjRootPrefix,
        bitcodeFile,
        bitcodeFiles,
        localLinkActionConstruction,
        featureConfiguration,
        toolchain,
        toolchain.getFdoContext(),
        usePicForLtoBackendActions,
        CcToolchainProvider.shouldCreatePerObjectDebugInfo(
            featureConfiguration, toolchain.getCppConfiguration()),
        argv);
  }

  private static ImmutableList<String> collectPerFileLtoBackendOpts(
      CppConfiguration cppConfiguration, Artifact objectFile) {
    return cppConfiguration.getPerFileLtoBackendOpts().stream()
        .filter(perLabelOptions -> perLabelOptions.isIncluded(objectFile))
        .map(PerLabelOptions::getOptions)
        .flatMap(options -> options.stream())
        .collect(toImmutableList());
  }

  private static List<String> getLtoBackendUserCompileFlags(
      CppConfiguration cppConfiguration, Artifact objectFile, ImmutableList<String> copts) {
    List<String> argv = new ArrayList<>();
    argv.addAll(cppConfiguration.getLinkopts());
    argv.addAll(copts);
    argv.addAll(cppConfiguration.getLtoBackendOptions());
    argv.addAll(collectPerFileLtoBackendOpts(cppConfiguration, objectFile));
    return argv;
  }

  public static ImmutableList<LtoBackendArtifacts> createLtoArtifacts(
      LinkActionConstruction linkActionConstruction,
      LtoCompilationContext ltoCompilationContext,
      FeatureConfiguration featureConfiguration,
      CcToolchainProvider toolchain,
      boolean usePicForLtoBackendActions,
      ImmutableList<Artifact> objectFiles,
      PathFragment ltoOutputRootPrefix,
      PathFragment ltoObjRootPrefix,
      List<LibraryToLink> staticLibrariesToLink,
      boolean allowLtoIndexing,
      boolean includeLinkStaticInLtoIndexing,
      boolean preferPicLibs)
      throws EvalException {
    Set<Artifact> compiled = new LinkedHashSet<>();
    Set<Artifact> staticLibraryArtifacts = new HashSet<>();
    for (LibraryToLink lib : staticLibrariesToLink) {
      boolean pic = lib.getEffectivePic(preferPicLibs);
      Artifact libArtifact = pic ? lib.getPicStaticLibrary() : lib.getStaticLibrary();
      if (!staticLibraryArtifacts.add(libArtifact)) {
        // Duplicated static libraries are linked just once and don't error out.
        // TODO(b/413333884): Clean up violations and error out
        continue;
      }
      LtoCompilationContext context =
          (pic ? lib.getPicLtoCompilationContext() : lib.getLtoCompilationContext());
      if (context != null) {
        compiled.addAll(context.getBitcodeFiles());
      }
    }

    // Make this a NestedSet to return from LtoBackendAction.getAllowedDerivedInputs. For M binaries
    // and N .o files, this is O(M*N). If we had nested sets of bitcode files, it would be O(M + N).
    NestedSetBuilder<Artifact> allBitcode = NestedSetBuilder.stableOrder();
    // Since this link includes object files from another library, we know that library must be
    // statically linked, so we need to look at includeLinkStaticInLtoIndexing to decide whether
    // to include its objects in the LTO indexing for this target.
    if (includeLinkStaticInLtoIndexing) {
      for (LibraryToLink lib : staticLibrariesToLink) {
        boolean pic = lib.getEffectivePic(preferPicLibs);
        ImmutableList<Artifact> libObjectFiles =
            pic ? lib.getPicObjectFiles() : lib.getObjectFiles();
        if (libObjectFiles == null) {
          continue;
        }
        for (Artifact objectFile : libObjectFiles) {
          if (compiled.contains(objectFile)) {
            allBitcode.add(objectFile);
          }
        }
      }
    }
    for (Artifact input : objectFiles) {
      if (ltoCompilationContext.containsBitcodeFile(input)) {
        allBitcode.add(input);
      }
    }
    BitcodeFiles bitcodeFiles = new BitcodeFiles(allBitcode.build());
    if (bitcodeFiles.getFiles().toList().stream().anyMatch(Artifact::isTreeArtifact)
        && ltoOutputRootPrefix.equals(ltoObjRootPrefix)) {
      throw Starlark.errorf(
          "Thinlto with tree artifacts requires feature use_lto_native_object_directory.");
    }
    ImmutableList.Builder<LtoBackendArtifacts> ltoOutputs = ImmutableList.builder();
    for (LibraryToLink lib : staticLibrariesToLink) {
      boolean pic = lib.getEffectivePic(preferPicLibs);
      ImmutableList<Artifact> libObjectFiles = pic ? lib.getPicObjectFiles() : lib.getObjectFiles();
      if (libObjectFiles == null) {
        continue;
      }
      LtoCompilationContext libLtoCompilationContext =
          pic ? lib.getPicLtoCompilationContext() : lib.getLtoCompilationContext();
      ImmutableMap<Artifact, LtoBackendArtifacts> sharedLtoBackends =
          pic ? lib.getPicSharedNonLtoBackends() : lib.getSharedNonLtoBackends();
      // We will create new LTO backends whenever we are performing LTO indexing, in which case
      // each target linking this library needs a unique set of LTO backends.
      for (Artifact objectFile : libObjectFiles) {
        if (compiled.contains(objectFile)) {
          if (includeLinkStaticInLtoIndexing) {
            List<String> backendUserCompileFlags =
                getLtoBackendUserCompileFlags(
                    toolchain.getCppConfiguration(),
                    objectFile,
                    libLtoCompilationContext.getCopts(objectFile));
            LtoBackendArtifacts ltoArtifacts =
                createLtoArtifact(
                    linkActionConstruction,
                    featureConfiguration,
                    toolchain,
                    usePicForLtoBackendActions,
                    objectFile,
                    bitcodeFiles,
                    ltoOutputRootPrefix,
                    ltoObjRootPrefix,
                    /* createSharedNonLto= */ false,
                    backendUserCompileFlags);
            ltoOutputs.add(ltoArtifacts);
          } else {
            // We should have created shared non-LTO backends when the library was created.
            if (sharedLtoBackends == null) {
              throw Starlark.errorf(
                  "Statically linked test target requires non-LTO backends for its library inputs,"
                      + " but library input %s does not specify shared_non_lto_backends",
                  lib);
            }
            LtoBackendArtifacts ltoArtifacts = sharedLtoBackends.getOrDefault(objectFile, null);
            Preconditions.checkNotNull(ltoArtifacts);
            ltoOutputs.add(ltoArtifacts);
          }
        }
      }
    }
    for (Artifact input : objectFiles) {
      if (ltoCompilationContext.containsBitcodeFile(input)) {
        List<String> backendUserCompileFlags =
            getLtoBackendUserCompileFlags(
                toolchain.getCppConfiguration(), input, ltoCompilationContext.getCopts(input));
        LtoBackendArtifacts ltoArtifacts =
            createLtoArtifact(
                linkActionConstruction,
                featureConfiguration,
                toolchain,
                usePicForLtoBackendActions,
                input,
                bitcodeFiles,
                ltoOutputRootPrefix,
                ltoObjRootPrefix,
                !allowLtoIndexing,
                backendUserCompileFlags);
        ltoOutputs.add(ltoArtifacts);
      }
    }

    return ltoOutputs.build();
  }

  public static ImmutableMap<Artifact, LtoBackendArtifacts> createSharedNonLtoArtifacts(
      LinkActionConstruction linkActionConstruction,
      LtoCompilationContext ltoCompilationContext,
      FeatureConfiguration featureConfiguration,
      CcToolchainProvider toolchain,
      boolean usePicForLtoBackendActions,
      ImmutableList<Artifact> objectFiles)
      throws EvalException {
    PathFragment ltoOutputRootPrefix = CppHelper.SHARED_NONLTO_BACKEND_ROOT_PREFIX;
    PathFragment ltoObjRootPrefix =
        featureConfiguration.isEnabled(CppRuleClasses.USE_LTO_NATIVE_OBJECT_DIRECTORY)
            ? CppHelper.getThinLtoNativeObjectDirectoryFromLtoOutputRoot(ltoOutputRootPrefix)
            : ltoOutputRootPrefix;

    ImmutableMap.Builder<Artifact, LtoBackendArtifacts> sharedNonLtoBackends =
        ImmutableMap.builder();

    for (Artifact inputArtifact : objectFiles) {
      if (ltoCompilationContext.containsBitcodeFile(inputArtifact)) {
        List<String> backendUserCompileFlags =
            getLtoBackendUserCompileFlags(
                toolchain.getCppConfiguration(),
                inputArtifact,
                ltoCompilationContext.getCopts(inputArtifact));
        LtoBackendArtifacts ltoArtifacts =
            createLtoArtifact(
                linkActionConstruction,
                featureConfiguration,
                toolchain,
                usePicForLtoBackendActions,
                inputArtifact,
                /* allBitcode= */ null,
                ltoOutputRootPrefix,
                ltoObjRootPrefix,
                /* createSharedNonLto= */ true,
                backendUserCompileFlags);
        sharedNonLtoBackends.put(inputArtifact, ltoArtifacts);
      }
    }

    return sharedNonLtoBackends.buildOrThrow();
  }

  public static ImmutableMap<Artifact, LtoBackendArtifacts> createSharedNonLtoArtifacts(
      LinkActionConstruction linkActionConstruction,
      LtoCompilationContext ltoCompilationContext,
      boolean isLinker,
      FeatureConfiguration featureConfiguration,
      CcToolchainProvider toolchain,
      boolean usePicForLtoBackendActions,
      ImmutableList<Artifact> objectFiles)
      throws EvalException {
    // Only create the shared LTO artifacts for a statically linked library that has bitcode files.
    if (ltoCompilationContext == null || isLinker) {
      return ImmutableMap.<Artifact, LtoBackendArtifacts>of();
    }

    return createSharedNonLtoArtifacts(
        linkActionConstruction,
        ltoCompilationContext,
        featureConfiguration,
        toolchain,
        usePicForLtoBackendActions,
        objectFiles);
  }
}
