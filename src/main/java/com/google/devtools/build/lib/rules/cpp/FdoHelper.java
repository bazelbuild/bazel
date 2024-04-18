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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.Tool;
import com.google.devtools.build.lib.rules.cpp.FdoContext.BranchFdoMode;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/** Helper responsible for creating {@link FdoContext} */
public class FdoHelper {

  @Nullable
  public static FdoContext getFdoContext(
      RuleContext ruleContext,
      BuildConfigurationValue configuration,
      CppConfiguration cppConfiguration,
      ImmutableMap<String, PathFragment> toolPaths,
      StructImpl fdoPrefetchProvider,
      StructImpl propellerOptimizeProvider,
      StructImpl memProfProfileProvider,
      StructImpl fdoOptimizeProvider,
      StructImpl fdoProfileProvider,
      StructImpl xFdoProfileProvider,
      StructImpl csFdoProfileProvider,
      NestedSet<Artifact> allFiles,
      Artifact zipper,
      CcToolchainConfigInfo ccToolchainConfigInfo,
      ImmutableList<Artifact> fdoOptimizeArtifacts,
      Label fdoOptimizeLabel)
      throws EvalException {
    FdoInputFile fdoInputFile = null;
    FdoInputFile csFdoInputFile = null;
    FdoInputFile prefetchHints = null;
    PropellerOptimizeInputFile propellerOptimizeInputFile = null;
    FdoInputFile memprofProfile = null;
    Artifact protoProfileArtifact = null;
    Pair<FdoInputFile, Artifact> fdoInputs = null;
    if (configuration.getCompilationMode() == CompilationMode.OPT) {
      if (cppConfiguration.getFdoPrefetchHintsLabel() != null) {
        prefetchHints = FdoInputFile.fromStarlarkProvider(fdoPrefetchProvider);
      }

      if (cppConfiguration.getPropellerOptimizeAbsoluteCCProfile() != null
          || cppConfiguration.getPropellerOptimizeAbsoluteLdProfile() != null) {
        Artifact ccArtifact = null;
        if (cppConfiguration.getPropellerOptimizeAbsoluteCCProfile() != null) {
          ccArtifact =
              PropellerOptimizeInputFile.createAbsoluteArtifact(
                  ruleContext, cppConfiguration.getPropellerOptimizeAbsoluteCCProfile());
        }
        Artifact ldArtifact = null;
        if (cppConfiguration.getPropellerOptimizeAbsoluteLdProfile() != null) {
          ldArtifact =
              PropellerOptimizeInputFile.createAbsoluteArtifact(
                  ruleContext, cppConfiguration.getPropellerOptimizeAbsoluteLdProfile());
        }
        if (ccArtifact != null || ldArtifact != null) {
          propellerOptimizeInputFile = new PropellerOptimizeInputFile(ccArtifact, ldArtifact);
        }
      } else if (cppConfiguration.getPropellerOptimizeLabel() != null) {
        propellerOptimizeInputFile =
            PropellerOptimizeInputFile.fromStarlarkProvider(propellerOptimizeProvider);
      }

      // Attempt to fetch the memprof profile input from an explicit flag or as part of the
      // fdo_profile rule. The former overrides the latter. Also handle the case where the
      // fdo_profile rule is specified using fdo_optimize.
      if (cppConfiguration.getMemProfProfileLabel() != null) {
        memprofProfile = FdoInputFile.fromStarlarkProvider(memProfProfileProvider);
      } else if (cppConfiguration.getFdoProfileLabel() != null
          && fdoProfileProvider.getValue("memprof_artifact") != Starlark.NONE) {
        memprofProfile =
            FdoInputFile.fromArtifact(
                fdoProfileProvider.getValue("memprof_artifact", Artifact.class));
      } else if (cppConfiguration.getFdoOptimizeLabel() != null
          && fdoOptimizeProvider != null
          && fdoOptimizeProvider.getValue("memprof_artifact") != Starlark.NONE) {
        memprofProfile =
            FdoInputFile.fromArtifact(
                fdoOptimizeProvider.getValue("memprof_artifact", Artifact.class));
      }

      if (cppConfiguration.getFdoPath() != null) {
        PathFragment fdoZip = cppConfiguration.getFdoPath();
        // fdoZip should be set if the profile is a path, fdoInputFile if it is an artifact, but
        // never both
        Preconditions.checkState(fdoInputFile == null);
        fdoInputFile = FdoInputFile.fromAbsolutePath(fdoZip);
      } else if (cppConfiguration.getFdoOptimizeLabel() != null) {
        if (fdoOptimizeProvider != null) {
          fdoInputs = getFdoInputs(ruleContext, fdoOptimizeProvider);
        } else {
          fdoInputFile =
              fdoInputFileFromArtifacts(ruleContext, fdoOptimizeArtifacts, fdoOptimizeLabel);
        }
      } else if (cppConfiguration.getFdoProfileLabel() != null) {
        fdoInputs = getFdoInputs(ruleContext, fdoProfileProvider);
      } else if (cppConfiguration.getXFdoProfileLabel() != null) {
        fdoInputs = getFdoInputs(ruleContext, xFdoProfileProvider);
      }

      Pair<FdoInputFile, Artifact> csFdoInputs = null;
      PathFragment csFdoZip = cppConfiguration.getCSFdoAbsolutePath();
      if (csFdoZip != null) {
        csFdoInputFile = FdoInputFile.fromAbsolutePath(csFdoZip);
      } else if (cppConfiguration.getCSFdoProfileLabel() != null) {
        csFdoInputs = getFdoInputs(ruleContext, csFdoProfileProvider);
      }
      if (csFdoInputs != null) {
        csFdoInputFile = csFdoInputs.getFirst();
      }
    }

    if (ruleContext.hasErrors()) {
      return null;
    }

    if (fdoInputs != null) {
      fdoInputFile = fdoInputs.getFirst();
      protoProfileArtifact = fdoInputs.getSecond();
    }

    FdoContext.BranchFdoProfile branchFdoProfile = null;
    if (fdoInputFile != null) {
      BranchFdoMode branchFdoMode;
      if (CppFileTypes.GCC_AUTO_PROFILE.matches(fdoInputFile)) {
        branchFdoMode = BranchFdoMode.AUTO_FDO;
      } else if (CppFileTypes.XBINARY_PROFILE.matches(fdoInputFile)) {
        branchFdoMode = BranchFdoMode.XBINARY_FDO;
      } else if (CppFileTypes.LLVM_PROFILE.matches(fdoInputFile)) {
        branchFdoMode = BranchFdoMode.LLVM_FDO;
      } else if (CppFileTypes.LLVM_PROFILE_RAW.matches(fdoInputFile)) {
        branchFdoMode = BranchFdoMode.LLVM_FDO;
      } else if (CppFileTypes.LLVM_PROFILE_ZIP.matches(fdoInputFile)) {
        branchFdoMode = BranchFdoMode.LLVM_FDO;
      } else {
        ruleContext.ruleError("invalid extension for FDO profile file.");
        return null;
      }
      // Check if this is LLVM_CS_FDO
      if (branchFdoMode == BranchFdoMode.LLVM_FDO) {
        if (csFdoInputFile != null) {
          branchFdoMode = BranchFdoMode.LLVM_CS_FDO;
        }
      }
      if ((branchFdoMode != BranchFdoMode.XBINARY_FDO)
          && (branchFdoMode != BranchFdoMode.AUTO_FDO)
          && cppConfiguration.getXFdoProfileLabel() != null) {
        throw Starlark.errorf("--xbinary_fdo only accepts *.xfdo and *.afdo");
      }

      if (configuration.isCodeCoverageEnabled()) {
        throw Starlark.errorf("coverage mode is not compatible with FDO optimization");
      }
      // This tries to convert LLVM profiles to the indexed format if necessary.
      Artifact profileArtifact = null;
      if (branchFdoMode == BranchFdoMode.LLVM_FDO) {
        profileArtifact =
            convertLLVMRawProfileToIndexed(
                fdoInputFile,
                toolPaths,
                ruleContext,
                "fdo",
                zipper,
                ccToolchainConfigInfo,
                allFiles);
        if (ruleContext.hasErrors()) {
          return null;
        }
      } else if (branchFdoMode == BranchFdoMode.AUTO_FDO
          || branchFdoMode == BranchFdoMode.XBINARY_FDO) {
        profileArtifact =
            ruleContext.getUniqueDirectoryArtifact(
                "fdo", fdoInputFile.getBasename(), ruleContext.getBinOrGenfilesDirectory());
        symlinkTo(
            ruleContext,
            profileArtifact,
            fdoInputFile,
            "Symlinking FDO profile " + fdoInputFile.getBasename());
      } else if (branchFdoMode == BranchFdoMode.LLVM_CS_FDO) {
        Artifact nonCSProfileArtifact =
            convertLLVMRawProfileToIndexed(
                fdoInputFile,
                toolPaths,
                ruleContext,
                "fdo",
                zipper,
                ccToolchainConfigInfo,
                allFiles);
        if (ruleContext.hasErrors()) {
          return null;
        }
        Artifact csProfileArtifact =
            convertLLVMRawProfileToIndexed(
                csFdoInputFile,
                toolPaths,
                ruleContext,
                "csfdo",
                zipper,
                ccToolchainConfigInfo,
                allFiles);
        if (ruleContext.hasErrors()) {
          return null;
        }
        if (nonCSProfileArtifact != null && csProfileArtifact != null) {
          profileArtifact =
              mergeLLVMProfiles(
                  allFiles,
                  toolPaths,
                  ruleContext,
                  nonCSProfileArtifact,
                  csProfileArtifact,
                  "mergedfdo",
                  "MergedCS.profdata");
          if (ruleContext.hasErrors()) {
            return null;
          }
        }
      }
      branchFdoProfile =
          new FdoContext.BranchFdoProfile(branchFdoMode, profileArtifact, protoProfileArtifact);
    }
    Artifact prefetchHintsArtifact = getPrefetchHintsArtifact(prefetchHints, ruleContext);

    Artifact memprofProfileArtifact =
        getMemProfProfileArtifact(zipper, memprofProfile, ruleContext);
    if (ruleContext.hasErrors()) {
      return null;
    }

    return new FdoContext(
        branchFdoProfile,
        prefetchHintsArtifact,
        propellerOptimizeInputFile,
        memprofProfileArtifact);
  }

  /**
   * Returns the profile name with the same file name as fdoProfile and an extension that matches
   * {@link FileType}.
   */
  private static String getLLVMProfileFileName(FdoInputFile fdoProfile, FileType type) {
    if (type.matches(fdoProfile)) {
      return fdoProfile.getBasename();
    } else {
      return FileSystemUtils.removeExtension(fdoProfile.getBasename())
          + type.getExtensions().get(0);
    }
  }

  @Nullable
  private static Artifact getPrefetchHintsArtifact(
      FdoInputFile prefetchHintsFile, RuleContext ruleContext) {
    if (prefetchHintsFile == null) {
      return null;
    }
    Artifact prefetchHintsArtifact = prefetchHintsFile.getArtifact();
    if (prefetchHintsArtifact != null) {
      return prefetchHintsArtifact;
    }

    prefetchHintsArtifact =
        ruleContext.getUniqueDirectoryArtifact(
            "fdo",
            prefetchHintsFile.getAbsolutePath().getBaseName(),
            ruleContext.getBinOrGenfilesDirectory());
    ruleContext.registerAction(
        SymlinkAction.toAbsolutePath(
            ruleContext.getActionOwner(),
            PathFragment.create(prefetchHintsFile.getAbsolutePath().getPathString()),
            prefetchHintsArtifact,
            "Symlinking LLVM Cache Prefetch Hints Profile "
                + prefetchHintsFile.getAbsolutePath().getPathString()));
    return prefetchHintsArtifact;
  }

  private static void symlinkTo(
      RuleContext ruleContext,
      Artifact symlink,
      FdoInputFile fdoInputFile,
      String progressMessage) {
    if (fdoInputFile.getArtifact() != null) {
      ruleContext.registerAction(
          SymlinkAction.toArtifact(
              ruleContext.getActionOwner(), fdoInputFile.getArtifact(), symlink, progressMessage));
    } else {
      ruleContext.registerAction(
          SymlinkAction.toAbsolutePath(
              ruleContext.getActionOwner(),
              fdoInputFile.getAbsolutePath(),
              symlink,
              progressMessage));
    }
  }

  /** This function merges profile1 and profile2 and generates mergedOutput. */
  private static Artifact mergeLLVMProfiles(
      NestedSet<Artifact> allFiles,
      ImmutableMap<String, PathFragment> toolPaths,
      RuleContext ruleContext,
      Artifact profile1,
      Artifact profile2,
      String fdoUniqueArtifactName,
      String mergedOutput) {
    Artifact profileArtifact =
        ruleContext.getUniqueDirectoryArtifact(
            fdoUniqueArtifactName, mergedOutput, ruleContext.getBinOrGenfilesDirectory());

    // Merge LLVM profiles.
    ruleContext.registerAction(
        new SpawnAction.Builder()
            .addInput(profile1)
            .addInput(profile2)
            .addTransitiveInputs(allFiles)
            .addOutput(profileArtifact)
            .useDefaultShellEnvironment()
            .setExecutable(toolPaths.get(Tool.LLVM_PROFDATA.getNamePart()))
            .setProgressMessage("LLVMProfDataAction: Generating %s", profileArtifact.prettyPrint())
            .setMnemonic("LLVMProfDataMergeAction")
            .addCommandLine(
                CustomCommandLine.builder()
                    .add("merge")
                    .add("-o")
                    .addExecPath(profileArtifact)
                    .addExecPath(profile1)
                    .addExecPath(profile2)
                    .build())
            .build(ruleContext));

    return profileArtifact;
  }

  /*
   * This function checks the format of the input profile data and converts it to
   * the indexed format (.profdata) if necessary.
   */
  @Nullable
  private static Artifact convertLLVMRawProfileToIndexed(
      FdoInputFile fdoProfile,
      ImmutableMap<String, PathFragment> toolPaths,
      RuleContext ruleContext,
      String fdoUniqueArtifactName,
      Artifact zipperBinaryArtifact,
      CcToolchainConfigInfo ccToolchainConfigInfo,
      NestedSet<Artifact> allFiles) {
    Artifact profileArtifact =
        ruleContext.getUniqueDirectoryArtifact(
            fdoUniqueArtifactName,
            getLLVMProfileFileName(fdoProfile, CppFileTypes.LLVM_PROFILE),
            ruleContext.getBinOrGenfilesDirectory());

    // If the profile file is already in the desired format, symlink to it and return.
    if (CppFileTypes.LLVM_PROFILE.matches(fdoProfile)) {
      symlinkTo(
          ruleContext,
          profileArtifact,
          fdoProfile,
          "Symlinking LLVM Profile " + fdoProfile.getBasename());
      return profileArtifact;
    }

    Artifact rawProfileArtifact;

    if (CppFileTypes.LLVM_PROFILE_ZIP.matches(fdoProfile)) {
      // Get the zipper binary for unzipping the profile.
      if (zipperBinaryArtifact == null) {
        ruleContext.ruleError(
            "Zipped profiles are not supported with platforms/toolchains before "
                + "toolchain-transitions are implemented.");
        return null;
      }

      // TODO(zhayu): find a way to avoid hard-coding cpu architecture here (b/65582760)
      String rawProfileFileName = "fdocontrolz_profile.profraw";
      String cpu = ccToolchainConfigInfo.getTargetCpu();
      if (!"k8".equals(cpu)) {
        rawProfileFileName = "fdocontrolz_profile-" + cpu + ".profraw";
      }
      rawProfileArtifact =
          ruleContext.getUniqueDirectoryArtifact(
              fdoUniqueArtifactName, rawProfileFileName, ruleContext.getBinOrGenfilesDirectory());

      // Symlink to the zipped profile file to extract the contents.
      Artifact zipProfileArtifact =
          ruleContext.getUniqueDirectoryArtifact(
              fdoUniqueArtifactName,
              fdoProfile.getBasename(),
              ruleContext.getBinOrGenfilesDirectory());
      symlinkTo(
          ruleContext,
          zipProfileArtifact,
          fdoProfile,
          "Symlinking LLVM ZIP Profile " + fdoProfile.getBasename());

      // We invoke different binaries depending on whether the unzip_fdo tool
      // is available. When it isn't, unzip_fdo is aliased to the generic
      // zipper tool, which takes different command-line arguments.
      CustomCommandLine.Builder argv = new CustomCommandLine.Builder();
      if (zipperBinaryArtifact.getExecPathString().endsWith("unzip_fdo")) {
        argv.addExecPath("--profile_zip", zipProfileArtifact)
            .add("--cpu", cpu)
            .add("--output_file", rawProfileArtifact.getExecPath().getSafePathString());
      } else {
        argv.addExecPath("xf", zipProfileArtifact)
            .add("-d", rawProfileArtifact.getExecPath().getParentDirectory().getSafePathString());
      }
      // Unzip the profile.
      ruleContext.registerAction(
          new SpawnAction.Builder()
              .addInput(zipProfileArtifact)
              .addInput(zipperBinaryArtifact)
              .addOutput(rawProfileArtifact)
              .useDefaultShellEnvironment()
              .setExecutable(zipperBinaryArtifact)
              .setProgressMessage(
                  "LLVMUnzipProfileAction: Generating %s", rawProfileArtifact.prettyPrint())
              .setMnemonic("LLVMUnzipProfileAction")
              .addCommandLine(argv.build())
              .build(ruleContext));
    } else {
      rawProfileArtifact =
          ruleContext.getUniqueDirectoryArtifact(
              fdoUniqueArtifactName,
              getLLVMProfileFileName(fdoProfile, CppFileTypes.LLVM_PROFILE_RAW),
              ruleContext.getBinOrGenfilesDirectory());
      symlinkTo(
          ruleContext,
          rawProfileArtifact,
          fdoProfile,
          "Symlinking LLVM Raw Profile " + fdoProfile.getBasename());
    }

    if (toolPaths.get(Tool.LLVM_PROFDATA.getNamePart()) == null) {
      ruleContext.ruleError(
          "llvm-profdata not available with this crosstool, needed for profile conversion");
      return null;
    }

    // Convert LLVM raw profile to indexed format.
    ruleContext.registerAction(
        new SpawnAction.Builder()
            .addInput(rawProfileArtifact)
            .addTransitiveInputs(allFiles)
            .addOutput(profileArtifact)
            .useDefaultShellEnvironment()
            .setExecutable(toolPaths.get(Tool.LLVM_PROFDATA.getNamePart()))
            .setProgressMessage("LLVMProfDataAction: Generating %s", profileArtifact.prettyPrint())
            .setMnemonic("LLVMProfDataAction")
            .addCommandLine(
                CustomCommandLine.builder()
                    .add("merge")
                    .add("-o")
                    .addExecPath(profileArtifact)
                    .addExecPath(rawProfileArtifact)
                    .build())
            .build(ruleContext));

    return profileArtifact;
  }

  /*
   * This function symlinks the memprof profile (after unzipping as needed).
   */
  @Nullable
  private static Artifact getMemProfProfileArtifact(
      Artifact zipperBinaryArtifact, FdoInputFile memprofProfile, RuleContext ruleContext) {
    if (memprofProfile == null) {
      return null;
    }
    String memprofUniqueArtifactName = "memprof";
    String memprofProfileFileName = "memprof.profdata";
    Artifact profileArtifact =
        ruleContext.getUniqueDirectoryArtifact(
            memprofUniqueArtifactName,
            memprofProfileFileName,
            ruleContext.getBinOrGenfilesDirectory());

    // If the profile file is already in the desired format, symlink to it and return.
    if (CppFileTypes.LLVM_PROFILE.matches(memprofProfile)) {
      symlinkTo(
          ruleContext,
          profileArtifact,
          memprofProfile,
          "Symlinking MemProf Profile " + memprofProfile.getBasename());
      return profileArtifact;
    }

    if (!CppFileTypes.LLVM_PROFILE_ZIP.matches(memprofProfile)) {
      ruleContext.ruleError("Expected zipped memprof profile.");
      return null;
    }

    // Get the zipper binary for unzipping the profile.
    if (zipperBinaryArtifact == null) {
      ruleContext.ruleError(
          "Zipped profiles are not supported with platforms/toolchains before "
              + "toolchain-transitions are implemented.");
      return null;
    }

    // Symlink to the zipped profile file to extract the contents.
    Artifact zipProfileArtifact =
        ruleContext.getUniqueDirectoryArtifact(
            memprofUniqueArtifactName,
            memprofProfile.getBasename(),
            ruleContext.getBinOrGenfilesDirectory());
    symlinkTo(
        ruleContext,
        zipProfileArtifact,
        memprofProfile,
        "Symlinking MemProf ZIP Profile " + memprofProfile.getBasename());

    CustomCommandLine.Builder argv = new CustomCommandLine.Builder();
    // We invoke different binaries depending on whether the unzip_fdo tool
    // is available. When it isn't, unzip_fdo is aliased to the generic
    // zipper tool, which takes different command-line arguments.
    if (zipperBinaryArtifact.getExecPathString().endsWith("unzip_fdo")) {
      argv.addExecPath("--profile_zip", zipProfileArtifact)
          .add("--memprof")
          .add("--output_file", profileArtifact.getExecPath().getSafePathString());
    } else {
      argv.addExecPath("xf", zipProfileArtifact)
          .add("-d", profileArtifact.getExecPath().getParentDirectory().getSafePathString());
    }
    // Unzip the profile.
    ruleContext.registerAction(
        new SpawnAction.Builder()
            .addInput(zipProfileArtifact)
            .addInput(zipperBinaryArtifact)
            .addOutput(profileArtifact)
            .useDefaultShellEnvironment()
            .setExecutable(zipperBinaryArtifact)
            .setProgressMessage("MemProfUnzipProfileAction: Generating %{output}")
            .setMnemonic("MemProfUnzipProfileAction")
            .addCommandLine(argv.build())
            .build(ruleContext));

    return profileArtifact;
  }

  @Nullable
  static Pair<FdoInputFile, Artifact> getFdoInputs(
      RuleContext ruleContext, StructImpl fdoProfileProvider) throws EvalException {
    if (fdoProfileProvider == null) {
      ruleContext.ruleError("--fdo_profile/--xbinary_fdo input needs to be an fdo_profile rule");
      return null;
    }
    return Pair.of(
        FdoInputFile.fromStarlarkProvider(fdoProfileProvider),
        fdoProfileProvider.getValue("proto_profile_artifact") == Starlark.NONE
            ? null
            : fdoProfileProvider.getValue("proto_profile_artifact", Artifact.class));
  }

  @Nullable
  private static FdoInputFile fdoInputFileFromArtifacts(
      RuleContext ruleContext,
      ImmutableList<Artifact> fdoOptimizeArtifacts,
      Label fdoOptimizeLabel) {
    if (fdoOptimizeArtifacts.size() != 1) {
      ruleContext.ruleError("--fdo_optimize does not point to a single target");
      return null;
    }

    Artifact fdoArtifact = fdoOptimizeArtifacts.get(0);
    if (!fdoOptimizeLabel
        .getPackageIdentifier()
        .getExecPath(ruleContext.getConfiguration().isSiblingRepositoryLayout())
        .getRelative(fdoOptimizeLabel.getName())
        .equals(fdoArtifact.getExecPath())) {
      ruleContext.ruleError(
          "--fdo_optimize points to a target that is not an input file or an fdo_profile rule");
      return null;
    }

    return FdoInputFile.fromArtifact(fdoArtifact);
  }
}
