// Copyright 2014 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import java.io.IOException;

/**
 * Support class for FDO (feedback directed optimization).
 *
 * <p>{@link FdoSupport#create} is called from {@link FdoSupportFunction} (a {@link SkyFunction}),
 * which is requested from Skyframe by the {@code cc_toolchain} rule.
 *
 * <p>For each C++ compile action in the target configuration, {@link #configureCompilation} is
 * called, which adds command line options and input files required for the build.
 */
@Immutable
@AutoCodec
public class FdoSupport {
  /**
   * The FDO mode we are operating in.
   */
  @VisibleForSerialization
  enum FdoMode {
    /** FDO is turned off. */
    OFF,

    /** Profiling-based FDO using an explicitly recorded profile. */
    VANILLA,

    /** FDO based on automatically collected data. */
    AUTO_FDO,

    /** FDO based on cross binary collected data. */
    XBINARY_FDO,

    /** Instrumentation-based FDO implemented on LLVM. */
    LLVM_FDO,
  }

  /**
   * Coverage information output directory passed to {@code --fdo_instrument},
   * or {@code null} if FDO instrumentation is disabled.
   */
  private final String fdoInstrument;

  /**
   * Path of the profile file passed to {@code --fdo_optimize}, or
   * {@code null} if FDO optimization is disabled.  The profile file
   * can be a coverage ZIP or an AutoFDO feedback file.
   */
  // TODO(lberki): this should be a PathFragment
  private final Path fdoProfile;

  /**
   * Temporary directory to which the coverage ZIP file is extracted to (relative to the exec root),
   * or {@code null} if FDO optimization is disabled. This is used to create artifacts for the
   * extracted files.
   *
   * <p>Note that this root is intentionally not registered with the artifact factory.
   */
  private final ArtifactRoot fdoRoot;

  /**
   * The relative path of the FDO root to the exec root.
   */
  private final PathFragment fdoRootExecPath;

  /**
   * Path of FDO files under the FDO root.
   */
  private final PathFragment fdoPath;

  /**
   * FDO mode.
   */
  private final FdoMode fdoMode;

  /**
   * Creates an FDO support object.
   *
   * @param fdoInstrument value of the --fdo_instrument option
   * @param fdoProfile path to the profile file passed to --fdo_optimize option
   */
  @VisibleForSerialization
  @AutoCodec.Instantiator
  FdoSupport(
      FdoMode fdoMode,
      ArtifactRoot fdoRoot,
      PathFragment fdoRootExecPath,
      String fdoInstrument,
      Path fdoProfile) {
    this.fdoInstrument = fdoInstrument;
    this.fdoProfile = fdoProfile;
    this.fdoRoot = fdoRoot;
    this.fdoRootExecPath = fdoRootExecPath;
    this.fdoPath = fdoProfile == null
        ? null
        : FileSystemUtils.removeExtension(PathFragment.create("_fdo").getChild(
            fdoProfile.getBaseName()));
    this.fdoMode = fdoMode;
  }

  public Path getFdoProfile() {
    return fdoProfile;
  }

  /** Creates an initialized {@link FdoSupport} instance. */
  static FdoSupport create(
      SkyFunction.Environment env,
      String fdoInstrument,
      Path fdoProfile,
      Path execRoot,
      String productName,
      FdoMode fdoMode) throws IOException, InterruptedException {

    ArtifactRoot fdoRoot =
        (fdoProfile == null)
            ? null
            : ArtifactRoot.asDerivedRoot(execRoot, execRoot.getRelative(productName + "-fdo"));

    PathFragment fdoRootExecPath = fdoProfile == null
        ? null
        : fdoRoot.getExecPath().getRelative(FileSystemUtils.removeExtension(
            PathFragment.create("_fdo").getChild(fdoProfile.getBaseName())));

    if (fdoProfile != null) {
        Path path = fdoMode == FdoMode.AUTO_FDO ? getAutoFdoImportsPath(fdoProfile) : fdoProfile;
        env.getValue(
            FileValue.key(
                RootedPath.toRootedPathMaybeUnderRoot(
                    path, ImmutableList.of(Root.fromPath(execRoot)))));

    }

    if (env.valuesMissing()) {
      return null;
    }

    if (fdoProfile != null && execRoot != null
        && (fdoMode == FdoMode.AUTO_FDO || fdoMode == FdoMode.XBINARY_FDO)) {
      FileSystemUtils.ensureSymbolicLink(
          execRoot.getRelative(getAutoProfilePath(fdoProfile, fdoRootExecPath)), fdoProfile);
    }
    return new FdoSupport(fdoMode, fdoRoot, fdoRootExecPath, fdoInstrument, fdoProfile);
  }

  private static Path getAutoFdoImportsPath(Path fdoProfile) {
     return fdoProfile.getParentDirectory().getRelative(fdoProfile.getBaseName() + ".imports");
  }

  /**
   * Configures a compile action builder by setting up command line options and auxiliary inputs
   * according to the FDO configuration. This method does nothing If FDO is disabled.
   */
  @ThreadSafe
  public ImmutableMap<String, String> configureCompilation(
      CppCompileActionBuilder builder,
      RuleContext ruleContext,
      FeatureConfiguration featureConfiguration,
      FdoSupportProvider fdoSupportProvider) {

    ImmutableMap.Builder<String, String> variablesBuilder = ImmutableMap.builder();

    if ((fdoSupportProvider != null)
        && (fdoSupportProvider.getPrefetchHintsArtifact() != null)) {
      variablesBuilder.put(
          CompileBuildVariables.FDO_PREFETCH_HINTS_PATH.getVariableName(),
          fdoSupportProvider.getPrefetchHintsArtifact().getExecPathString());
    }

    // FDO is disabled -> do nothing.
    if ((fdoInstrument == null) && (fdoRoot == null)) {
      return ImmutableMap.of();
    }

    if (featureConfiguration.isEnabled(CppRuleClasses.FDO_INSTRUMENT)) {
      variablesBuilder.put(
          CompileBuildVariables.FDO_INSTRUMENT_PATH.getVariableName(), fdoInstrument);
    }

    // Optimization phase
    if (fdoRoot != null) {
      AnalysisEnvironment env = ruleContext.getAnalysisEnvironment();
      // Declare dependency on contents of zip file.
      if (env.getSkyframeEnv().valuesMissing()) {
        return ImmutableMap.of();
      }
      Iterable<Artifact> auxiliaryInputs =
          getAuxiliaryInputs(fdoSupportProvider);
      builder.addMandatoryInputs(auxiliaryInputs);
      if (!Iterables.isEmpty(auxiliaryInputs)) {
        if (featureConfiguration.isEnabled(CppRuleClasses.AUTOFDO)
            || featureConfiguration.isEnabled(CppRuleClasses.XBINARYFDO)) {
          variablesBuilder.put(
              CompileBuildVariables.FDO_PROFILE_PATH.getVariableName(),
              getAutoProfilePath(fdoProfile, fdoRootExecPath).getPathString());
        }
        if (featureConfiguration.isEnabled(CppRuleClasses.FDO_OPTIMIZE)) {
          if (fdoMode == FdoMode.LLVM_FDO) {
            variablesBuilder.put(
                CompileBuildVariables.FDO_PROFILE_PATH.getVariableName(),
                fdoSupportProvider.getProfileArtifact().getExecPathString());
          } else {
            variablesBuilder.put(
                CompileBuildVariables.FDO_PROFILE_PATH.getVariableName(),
                fdoRootExecPath.getPathString());
          }
        }
      }
    }
    return variablesBuilder.build();
  }

  /** Returns the auxiliary files that need to be added to the {@link CppCompileAction}. */
  private Iterable<Artifact> getAuxiliaryInputs(FdoSupportProvider fdoSupportProvider) {
    ImmutableSet.Builder<Artifact> auxiliaryInputs = ImmutableSet.builder();

    if (fdoSupportProvider.getPrefetchHintsArtifact() != null) {
      auxiliaryInputs.add(fdoSupportProvider.getPrefetchHintsArtifact());
    }
    // If --fdo_optimize was not specified, we don't have any additional inputs.
    if (fdoProfile == null) {
      return auxiliaryInputs.build();
    } else if (fdoMode == FdoMode.LLVM_FDO
        || fdoMode == FdoMode.AUTO_FDO
        || fdoMode == FdoMode.XBINARY_FDO) {
      auxiliaryInputs.add(fdoSupportProvider.getProfileArtifact());
      return auxiliaryInputs.build();
    } else {
      return auxiliaryInputs.build();
    }
  }

  private static PathFragment getAutoProfilePath(Path fdoProfile, PathFragment fdoRootExecPath) {
    return fdoRootExecPath.getRelative(getAutoProfileRootRelativePath(fdoProfile));
  }

  private static PathFragment getAutoProfileRootRelativePath(Path fdoProfile) {
    return PathFragment.create(fdoProfile.getBaseName());
  }

  /**
   * Returns whether AutoFDO is enabled.
   */
  @ThreadSafe
  public boolean isAutoFdoEnabled() {
    return fdoMode == FdoMode.AUTO_FDO;
  }

  /** Returns whether crossbinary FDO is enabled. */
  @ThreadSafe
  public boolean isXBinaryFdoEnabled() {
    return fdoMode == FdoMode.XBINARY_FDO;
  }

  /**
   * Adds the FDO profile output path to the variable builder. If FDO is disabled, no build variable
   * is added.
   */
  @ThreadSafe
  public void getLinkOptions(
      FeatureConfiguration featureConfiguration, CcToolchainVariables.Builder buildVariables) {
    if (featureConfiguration.isEnabled(CppRuleClasses.FDO_INSTRUMENT)) {
      buildVariables.addStringVariable("fdo_instrument_path", fdoInstrument);
    }
  }

  /**
   * Adds the AutoFDO profile path to the variable builder and returns the profile artifact. If
   * AutoFDO is disabled, no build variable is added and returns null.
   */
  @ThreadSafe
  public ProfileArtifacts buildProfileForLtoBackend(
      FdoSupportProvider fdoSupportProvider,
      FeatureConfiguration featureConfiguration,
      CcToolchainVariables.Builder buildVariables) {
    Artifact prefetch = fdoSupportProvider.getPrefetchHintsArtifact();
    if (prefetch != null) {
      buildVariables.addStringVariable("fdo_prefetch_hints_path", prefetch.getExecPathString());
    }
    if (!featureConfiguration.isEnabled(CppRuleClasses.AUTOFDO)
        && !featureConfiguration.isEnabled(CppRuleClasses.XBINARYFDO)) {
      return new ProfileArtifacts(null, prefetch);
    }

    Artifact profile = fdoSupportProvider.getProfileArtifact();
    buildVariables.addStringVariable("fdo_profile_path", profile.getExecPathString());
    return new ProfileArtifacts(profile, prefetch);
  }

  public FdoSupportProvider createFdoSupportProvider(
      RuleContext ruleContext, ProfileArtifacts profiles) {
    if (fdoRoot == null) {
      return new FdoSupportProvider(this, profiles);
    }

    if (fdoMode == FdoMode.LLVM_FDO) {
      Preconditions.checkState(profiles != null && profiles.getProfileArtifact() != null);
      return new FdoSupportProvider(this, profiles);
    }

    Preconditions.checkState(fdoPath != null);
    PathFragment profileRootRelativePath = getAutoProfileRootRelativePath(fdoProfile);

    Artifact profileArtifact =
        ruleContext
            .getAnalysisEnvironment()
            .getDerivedArtifact(fdoPath.getRelative(profileRootRelativePath), fdoRoot);
    ruleContext.registerAction(new FdoStubAction(ruleContext.getActionOwner(), profileArtifact));
    Preconditions.checkState(fdoPath != null);

    return new FdoSupportProvider(
        this,
        new ProfileArtifacts(
            profileArtifact, profiles == null ? null : profiles.getPrefetchHintsArtifact()));
  }
}
