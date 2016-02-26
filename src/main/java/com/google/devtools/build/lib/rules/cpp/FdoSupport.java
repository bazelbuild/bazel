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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadHostile;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.ZipFileSystem;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.LipoMode;
import com.google.devtools.build.skyframe.SkyFunction;

import java.io.IOException;
import java.util.Collection;
import java.util.zip.ZipException;

/**
 * Support class for FDO (feedback directed optimization) and LIPO (lightweight inter-procedural
 * optimization).
 *
 * <p>There is a 1:1 relationship between {@link CppConfiguration} objects and {@code FdoSupport}
 * objects. The FDO support of a build configuration can be retrieved using {@link
 * CppConfiguration#getFdoSupport()}.
 *
 * <p>With respect to thread-safety, the {@link #prepareToBuild} method is not thread-safe, and must
 * not be called concurrently with other methods on this class.
 *
 * <p>Here follows a quick run-down of how FDO/LIPO builds work (for non-FDO/LIPO builds, none
 * of this applies):
 *
 * <p>{@link CppConfiguration#prepareHook} is called before the analysis phase, which calls
 * {@link #prepareToBuild}, which extracts the FDO .zip (in case we work with an explicitly
 * generated FDO profile file) or analyzes the .afdo.imports file next to the .afdo file (if
 * AutoFDO is in effect).
 *
 * <p>.afdo.imports files contain one import a line. A line is two paths separated by a colon,
 * with functions in the second path being referenced by functions in the first path. These are
 * then put into the imports map. If we do AutoFDO, we don't handle individual .gcda files, so
 * gcdaFiles will be empty.
 *
 * <p>Regular .fdo zip files contain .gcda files (which are added to gcdaFiles) and
 * .gcda.imports files. There is one .gcda.imports file for every source file and it contains one
 * path in every line, which can either be a path to a source file that contains a function
 * referenced by the original source file or the .gcda file for such a referenced file. They
 * both are added to the imports map.
 *
 * <p>If we do LIPO, we create an extra configuration that is called the "LIPO context collector",
 * whose job it is to collect information that every configured target compiled with LIPO needs.
 * The top-level target of this configuration is the LIPO context (always a cc_binary) and is an
 * implicit dependency of every cc_* rule through their :lipo_context_collector attribute. The
 * collected information is encapsulated in {@link LipoContextProvider}.
 *
 * <p>For each C++ compile action in the target configuration, {@link #configureCompilation} is
 * called, which adds command line options and input files required for the build. There are
 * three cases:
 *
 * <ul>
 * <li>If we do AutoFDO, the .afdo file and the source files containing the functions imported
 * by the original source file (as determined from the inputs map) are added.
 * <li>If we do FDO, the .gcda file corresponding to the source file is added.
 * <li>If we do LIPO, in addition to the .gcda file corresponding to the source file
 * (like for FDO) the source files that contain the functions referenced by the source file and
 * their .gcda files are added, too.
 * </ul>
 *
 * <p>If we do LIPO, the actual C++ compilation context for LIPO compilation actions is pieced
 * together from the CppCompileContext in LipoContextProvider and that of the rule being compiled.
 * (see {@link CppCompilationContext#mergeForLipo}) This is so that the include files for the
 * extra LIPO sources are found and is, strictly speaking, incorrect, since it also changes the
 * declared include directories of the main source file, which in theory can result in the
 * compilation passing even though it should fail with undeclared inclusion errors.
 *
 * <p>During the actual execution of the C++ compile action, the extra sources also need to be
 * include scanned, which is the reason why they are {@link IncludeScannable} objects and not
 * simple artifacts. We currently create these {@link IncludeScannable} objects by creating actual
 * C++ compile actions in the LIPO context collector configuration which are then never executed.
 * In fact, these C++ compile actions are never even registered with Skyframe. For this we
 * propagate a bit from {@code BuildConfiguration.isActionsEnabled} to
 * {@code CachingAnalysisEnvironment.allowRegisteringActions}, which causes actions to be silently
 * discarded after configured targets are created.
 */
public class FdoSupport {

  /**
   * Path within profile data .zip files that is considered the root of the
   * profile information directory tree.
   */
  private static final PathFragment ZIP_ROOT = new PathFragment("/");

  /**
   * Returns true if the given fdoFile represents an AutoFdo profile.
   */
  public static final boolean isAutoFdo(String fdoFile) {
    return CppFileTypes.GCC_AUTO_PROFILE.matches(fdoFile);
  }

  /**
   * Returns true if the given fdoFile represents an LLVM profile.
   */
  public static final boolean isLLVMFdo(String fdoFile) {
    return CppFileTypes.LLVM_PROFILE.matches(fdoFile);
  }

  /**
   * Coverage information output directory passed to {@code --fdo_instrument},
   * or {@code null} if FDO instrumentation is disabled.
   */
  private final PathFragment fdoInstrument;

  /**
   * Path of the profile file passed to {@code --fdo_optimize}, or
   * {@code null} if FDO optimization is disabled.  The profile file
   * can be a coverage ZIP or an AutoFDO feedback file.
   */
  private final Path fdoProfile;

  /**
   * Temporary directory to which the coverage ZIP file is extracted to
   * (relative to the exec root), or {@code null} if FDO optimization is
   * disabled. This is used to create artifacts for the extracted files.
   *
   * <p>Note that this root is intentionally not registered with the artifact
   * factory.
   */
  private final Root fdoRoot;

  /**
   * The relative path of the FDO root to the exec root.
   */
  private final PathFragment fdoRootExecPath;

  /**
   * Path of FDO files under the FDO root.
   */
  private final PathFragment fdoPath;

  /**
   * LIPO mode passed to {@code --lipo}. This is only used if
   * {@code fdoProfile != null}.
   */
  private final LipoMode lipoMode;

  /**
   * Flag indicating whether to use AutoFDO (as opposed to
   * instrumentation-based FDO).
   */
  private final boolean useAutoFdo;

  /**
   * Flag indicating whether to use LLVM instrumentation-based FDO (as
   * opposed to GCC instrumentation-based FDO).
   */
  private final boolean useLLVMFdo;

  /**
   * The {@code .gcda} files that have been extracted from the ZIP file,
   * relative to the root of the ZIP file.
   *
   * <p>Set only in {@link #prepareToBuild}.
   */
  private ImmutableSet<PathFragment> gcdaFiles = ImmutableSet.of();

  /**
   * Multimap from .gcda file base names to auxiliary input files.
   *
   * <p>The keys of the multimap are the exec root relative paths of .gcda files
   * with the extension removed. The values are the lines from the accompanying
   * .gcda.imports file.
   *
   * <p>The contents of the multimap are copied verbatim from the .gcda.imports
   * files and not yet checked for validity.
   *
   * <p>Set only in {@link #prepareToBuild}.
   */
  private ImmutableMultimap<PathFragment, PathFragment> imports;

  /**
   * Creates an FDO support object.
   *
   * @param fdoInstrument value of the --fdo_instrument option
   * @param fdoProfile path to the profile file passed to --fdo_optimize option
   * @param lipoMode value of the --lipo_mode option
   */
  public FdoSupport(PathFragment fdoInstrument, Path fdoProfile, LipoMode lipoMode, Path execRoot) {
    this.fdoInstrument = fdoInstrument;
    this.fdoProfile = fdoProfile;
    this.fdoRoot = (fdoProfile == null)
        ? null
        : Root.asDerivedRoot(execRoot, execRoot.getRelative("blaze-fdo"));
    this.fdoRootExecPath = fdoProfile == null
        ? null
        : fdoRoot.getExecPath().getRelative(FileSystemUtils.removeExtension(
            new PathFragment("_fdo").getChild(fdoProfile.getBaseName())));
    this.fdoPath = fdoProfile == null
        ? null
        : FileSystemUtils.removeExtension(new PathFragment("_fdo").getChild(
            fdoProfile.getBaseName()));
    this.lipoMode = lipoMode;
    this.useAutoFdo = fdoProfile != null && isAutoFdo(fdoProfile.getBaseName());
    this.useLLVMFdo = fdoProfile != null && isLLVMFdo(fdoProfile.getBaseName());
  }

  public Root getFdoRoot() {
    return fdoRoot;
  }

  public void declareSkyframeDependencies(SkyFunction.Environment env, Path execRoot) {
    if (fdoProfile != null) {
      if (isLipoEnabled()) {
        // Incrementality is not supported for LIPO builds, see FdoSupport#scannables.
        // Ensure that the Skyframe value containing the configuration will not be reused to avoid
        // incrementality issues.
        PrecomputedValue.dependOnBuildId(env);
        return;
      }

      // IMPORTANT: Keep the following in sync with #prepareToBuild.
      Path path;
      if (useAutoFdo) {
        path = fdoProfile.getParentDirectory().getRelative(
            fdoProfile.getBaseName() + ".imports");
      } else {
        path = fdoProfile;
      }
      env.getValue(FileValue.key(RootedPath.toRootedPathMaybeUnderRoot(path,
          ImmutableList.of(execRoot))));
    }
  }

  /**
   * Prepares the FDO support for building.
   *
   * <p>When an {@code --fdo_optimize} compile is requested, unpacks the given
   * FDO gcda zip file into a clean working directory under execRoot.
   *
   * @throws FdoException if the FDO ZIP contains a file of unknown type
   */
  @ThreadHostile // must be called before starting the build
  public void prepareToBuild(Path execRoot)
      throws IOException, FdoException {
    // The execRoot != null case is only there for testing. We cannot provide a real ZIP file in
    // tests because ZipFileSystem does not work with a ZIP on an in-memory file system.
    // IMPORTANT: Keep in sync with #declareSkyframeDependencies to avoid incrementality issues.
    if (fdoProfile != null && execRoot != null) {
      Path fdoDirPath = execRoot.getRelative(fdoRootExecPath);

      FileSystemUtils.deleteTreesBelow(fdoDirPath);
      FileSystemUtils.createDirectoryAndParents(fdoDirPath);

      if (useAutoFdo) {
        Path fdoImports = fdoProfile.getParentDirectory().getRelative(
            fdoProfile.getBaseName() + ".imports");
        if (isLipoEnabled()) {
          imports = readAutoFdoImports(fdoImports);
        }
        FileSystemUtils.ensureSymbolicLink(
            execRoot.getRelative(getAutoProfilePath()), fdoProfile);
      } else if (useLLVMFdo) {
        FileSystemUtils.ensureSymbolicLink(
            execRoot.getRelative(getLLVMProfilePath()), fdoProfile);
      } else {
        Path zipFilePath = new ZipFileSystem(fdoProfile).getRootDirectory();
        if (!zipFilePath.getRelative("blaze-out").isDirectory()) {
          throw new ZipException("FDO zip files must be zipped directly above 'blaze-out' " +
                                 "for the compiler to find the profile");
        }
        ImmutableSet.Builder<PathFragment> gcdaFilesBuilder = ImmutableSet.builder();
        ImmutableMultimap.Builder<PathFragment, PathFragment> importsBuilder =
            ImmutableMultimap.builder();
        extractFdoZip(zipFilePath, fdoDirPath, gcdaFilesBuilder, importsBuilder);
        gcdaFiles = gcdaFilesBuilder.build();
        imports = importsBuilder.build();
      }
    }
  }

  /**
   * Recursively extracts a directory from the GCDA ZIP file into a target
   * directory.
   *
   * <p>Imports files are not written to disk. Their content is directly added
   * to an internal data structure.
   *
   * <p>The files are written at $EXECROOT/blaze-fdo/_fdo/(base name of profile zip), and the
   * {@code _fdo} directory there is symlinked to from the exec root, so that the file are also
   * available at $EXECROOT/_fdo/..., which is their exec path. We need to jump through these
   * hoops because the FDO root 1. needs to be a source root, thus the exec path of its root is
   * ".", 2. it must not be equal to the exec root so that the artifact factory does not get
   * confused, 3. the files under it must be reachable by their exec path from the exec root.
   *
   * @throws IOException if any of the I/O operations failed
   * @throws FdoException if the FDO ZIP contains a file of unknown type
   */
  private void extractFdoZip(Path sourceDir, Path targetDir,
      ImmutableSet.Builder<PathFragment> gcdaFilesBuilder,
      ImmutableMultimap.Builder<PathFragment, PathFragment> importsBuilder)
          throws IOException, FdoException {
    for (Path sourceFile : sourceDir.getDirectoryEntries()) {
      Path targetFile = targetDir.getRelative(sourceFile.getBaseName());
      if (sourceFile.isDirectory()) {
        targetFile.createDirectory();
        extractFdoZip(sourceFile, targetFile, gcdaFilesBuilder, importsBuilder);
      } else {
        if (CppFileTypes.COVERAGE_DATA.matches(sourceFile)) {
          FileSystemUtils.copyFile(sourceFile, targetFile);
          gcdaFilesBuilder.add(
              sourceFile.relativeTo(sourceFile.getFileSystem().getRootDirectory()));
        } else if (CppFileTypes.COVERAGE_DATA_IMPORTS.matches(sourceFile)) {
          readCoverageImports(sourceFile, importsBuilder);
        } else {
            throw new FdoException("FDO ZIP file contained a file of unknown type: " + sourceFile);
        }
      }
    }
  }

  /**
   * Reads a .gcda.imports file and stores the imports information.
   *
   * @throws FdoException if an auxiliary LIPO input was not found
   */
  private void readCoverageImports(Path importsFile,
      ImmutableMultimap.Builder<PathFragment, PathFragment> importsBuilder)
          throws IOException, FdoException {
    PathFragment key = importsFile.asFragment().relativeTo(ZIP_ROOT);
    String baseName = key.getBaseName();
    String ext = Iterables.getOnlyElement(CppFileTypes.COVERAGE_DATA_IMPORTS.getExtensions());
    key = key.replaceName(baseName.substring(0, baseName.length() - ext.length()));

    for (String line : FileSystemUtils.iterateLinesAsLatin1(importsFile)) {
      if (!line.isEmpty()) {
        // We can't yet fully check the validity of a line. this is done later
        // when we actually parse the contained paths.
        PathFragment execPath = new PathFragment(line);
        if (execPath.isAbsolute()) {
          throw new FdoException("Absolute paths not allowed in gcda imports file " + importsFile
              + ": " + execPath);
        }
        importsBuilder.put(key, new PathFragment(line));
      }
    }
  }

  /**
   * Reads a .afdo.imports file and stores the imports information.
   */
  private ImmutableMultimap<PathFragment, PathFragment> readAutoFdoImports(Path importsFile)
          throws IOException, FdoException {
    ImmutableMultimap.Builder<PathFragment, PathFragment> importBuilder =
        ImmutableMultimap.builder();
    for (String line : FileSystemUtils.iterateLinesAsLatin1(importsFile)) {
      if (!line.isEmpty()) {
        PathFragment key = new PathFragment(line.substring(0, line.indexOf(':')));
        if (key.isAbsolute()) {
          throw new FdoException("Absolute paths not allowed in afdo imports file " + importsFile
              + ": " + key);
        }
        for (String auxFile : line.substring(line.indexOf(':') + 1).split(" ")) {
          if (auxFile.length() == 0) {
            continue;
          }

          importBuilder.put(key, new PathFragment(auxFile));
        }
      }
    }
    return importBuilder.build();
  }

  /**
   * Returns the imports from the .afdo.imports file of a source file.
   *
   * @param sourceExecPath the source file
   */
  private Collection<Artifact> getAutoFdoImports(RuleContext ruleContext,
      PathFragment sourceExecPath, LipoContextProvider lipoContextProvider) {
    Preconditions.checkState(isLipoEnabled());
    ImmutableCollection<PathFragment> afdoImports = imports.get(sourceExecPath);
    Preconditions.checkState(afdoImports != null,
        "AutoFDO import data missing for %s", sourceExecPath);
    ImmutableList.Builder<Artifact> result = ImmutableList.builder();
    for (PathFragment afdoImport : afdoImports) {
      Artifact afdoArtifact = lipoContextProvider.getSourceArtifactMap().get(afdoImport);
      if (afdoArtifact != null) {
        result.add(afdoArtifact);
      } else {
        ruleContext.ruleError(String.format(
            "cannot find source file '%s' referenced from '%s'", afdoImport, sourceExecPath));
      }
    }
    return result.build();
  }

  /**
   * Returns the imports from the .gcda.imports file of an object file.
   *
   * @param objDirectory the object directory of the object file's target
   * @param objectName the object file
   */
  private Iterable<PathFragment> getImports(PathFragment objDirectory, PathFragment objectName) {
    Preconditions.checkState(isLipoEnabled());
    Preconditions.checkState(imports != null,
        "Tried to look up imports of uninitialized FDOSupport");
    PathFragment key = objDirectory.getRelative(FileSystemUtils.removeExtension(objectName));
    ImmutableCollection<PathFragment> importsForObject = imports.get(key);
    Preconditions.checkState(importsForObject != null, "Import data missing for %s", key);
    return importsForObject;
  }

  /**
   * Configures a compile action builder by setting up command line options and
   * auxiliary inputs according to the FDO configuration. This method does
   * nothing If FDO is disabled.
   */
  @ThreadSafe
  public void configureCompilation(CppCompileActionBuilder builder,
      CcToolchainFeatures.Variables.Builder buildVariables, RuleContext ruleContext,
      PathFragment sourceName, PathFragment sourceExecPath, boolean usePic,
      FeatureConfiguration featureConfiguration) {
    // It is a bug if this method is called with useLipo if lipo is disabled. However, it is legal
    // if is is called with !useLipo, even though lipo is enabled.
    LipoContextProvider lipoInputProvider = CppHelper.getLipoContextProvider(ruleContext);
    Preconditions.checkArgument(lipoInputProvider == null || isLipoEnabled());

    // FDO is disabled -> do nothing.
    if ((fdoInstrument == null) && (fdoRoot == null)) {
      return;
    }

    if (featureConfiguration.isEnabled(CppRuleClasses.FDO_INSTRUMENT)) {
      buildVariables.addVariable("fdo_instrument_path",
          fdoInstrument.getPathString());
    }

    // Optimization phase
    if (fdoRoot != null) {
      AnalysisEnvironment env = ruleContext.getAnalysisEnvironment();
      // Declare dependency on contents of zip file.
      if (env.getSkyframeEnv().valuesMissing()) {
        return;
      }
      Iterable<Artifact> auxiliaryInputs = getAuxiliaryInputs(
          ruleContext, env, sourceName, sourceExecPath, usePic, lipoInputProvider);
      builder.addMandatoryInputs(auxiliaryInputs);
      if (!Iterables.isEmpty(auxiliaryInputs)) {
        if (featureConfiguration.isEnabled(CppRuleClasses.AUTOFDO)) {
          buildVariables.addVariable("fdo_profile_path",
              getAutoProfilePath().getPathString());
        }
        if (featureConfiguration.isEnabled(CppRuleClasses.FDO_OPTIMIZE)) {
          if (useLLVMFdo) {
            buildVariables.addVariable("fdo_profile_path",
                getLLVMProfilePath().getPathString());
          } else {
            buildVariables.addVariable("fdo_profile_path",
                fdoRootExecPath.getPathString());
          }
        }
      }
    }
  }

  /**
   * Returns the auxiliary files that need to be added to the {@link CppCompileAction}.
   */
  private Iterable<Artifact> getAuxiliaryInputs(
      RuleContext ruleContext, AnalysisEnvironment env, PathFragment sourceName,
      PathFragment sourceExecPath, boolean usePic, LipoContextProvider lipoContextProvider) {
    // If --fdo_optimize was not specified, we don't have any additional inputs.
    if (fdoProfile == null) {
      return ImmutableSet.of();
    } else if (useAutoFdo || useLLVMFdo) {
      ImmutableSet.Builder<Artifact> auxiliaryInputs = ImmutableSet.builder();

      PathFragment profileRootRelativePath = (useLLVMFdo)
          ? getLLVMProfileRootRelativePath()
          : getAutoProfileRootRelativePath();
      Artifact artifact = env.getDerivedArtifact(
          fdoPath.getRelative(profileRootRelativePath), fdoRoot);
      env.registerAction(new FdoStubAction(ruleContext.getActionOwner(), artifact));
      auxiliaryInputs.add(artifact);
      if (lipoContextProvider != null) {
        auxiliaryInputs.addAll(getAutoFdoImports(ruleContext, sourceExecPath, lipoContextProvider));
      }
      return auxiliaryInputs.build();
    } else {
      ImmutableSet.Builder<Artifact> auxiliaryInputs = ImmutableSet.builder();

      PathFragment objectName =
          FileSystemUtils.replaceExtension(sourceName, usePic ? ".pic.o" : ".o");

      Label lipoLabel = ruleContext.getLabel();
      auxiliaryInputs.addAll(
          getGcdaArtifactsForObjectFileName(ruleContext, env, objectName, lipoLabel));

      if (lipoContextProvider != null) {
        for (PathFragment importedFile : getImports(
            getNonLipoObjDir(ruleContext, lipoLabel), objectName)) {
          if (CppFileTypes.COVERAGE_DATA.matches(importedFile.getBaseName())) {
            Artifact gcdaArtifact = getGcdaArtifactsForGcdaPath(ruleContext, env, importedFile);
            if (gcdaArtifact == null) {
              ruleContext.ruleError(String.format(
                  ".gcda file %s is not in the FDO zip (referenced by source file %s)",
                  importedFile, sourceName));
            } else {
              auxiliaryInputs.add(gcdaArtifact);
            }
          } else {
            Artifact importedArtifact = lipoContextProvider.getSourceArtifactMap()
                .get(importedFile);
            if (importedArtifact != null) {
              auxiliaryInputs.add(importedArtifact);
            } else {
              ruleContext.ruleError(String.format(
                  "cannot find source file '%s' referenced from '%s'", importedFile, objectName));
            }
          }
        }
      }

      return auxiliaryInputs.build();
    }
  }

  /**
   * Returns the .gcda file artifacts for a .gcda path from the .gcda.imports file or null if the
   * referenced .gcda file is not in the FDO zip.
   */
  private Artifact getGcdaArtifactsForGcdaPath(RuleContext ruleContext,
      AnalysisEnvironment env, PathFragment gcdaPath) {
    if (!gcdaFiles.contains(gcdaPath)) {
      return null;
    }

    Artifact artifact = env.getDerivedArtifact(fdoPath.getRelative(gcdaPath), fdoRoot);
    env.registerAction(new FdoStubAction(ruleContext.getActionOwner(), artifact));
    return artifact;
  }

  private PathFragment getNonLipoObjDir(RuleContext ruleContext, Label label) {
    return ruleContext.getConfiguration().getBinFragment()
        .getRelative(CppHelper.getObjDirectory(label));
  }

  /**
   * Returns a list of .gcda file artifacts for an object file path.
   *
   * <p>The resulting set is either empty (because no .gcda file exists for the
   * given object file) or contains one or two artifacts (the file itself and a
   * symlink to it).
   */
  private ImmutableList<Artifact> getGcdaArtifactsForObjectFileName(RuleContext ruleContext,
      AnalysisEnvironment env, PathFragment objectFileName, Label lipoLabel) {
    // We put the .gcda files relative to the location of the .o file in the instrumentation run.
    String gcdaExt = Iterables.getOnlyElement(CppFileTypes.COVERAGE_DATA.getExtensions());
    PathFragment baseName = FileSystemUtils.replaceExtension(objectFileName, gcdaExt);
    PathFragment gcdaFile = getNonLipoObjDir(ruleContext, lipoLabel).getRelative(baseName);

    if (!gcdaFiles.contains(gcdaFile)) {
      // If the object is a .pic.o file and .pic.gcda is not found, we should try finding .gcda too
      String picoExt = Iterables.getOnlyElement(CppFileTypes.PIC_OBJECT_FILE.getExtensions());
      baseName = FileSystemUtils.replaceExtension(objectFileName, gcdaExt, picoExt);
      if (baseName == null) {
        // Object file is not .pic.o
        return ImmutableList.of();
      }
      gcdaFile = getNonLipoObjDir(ruleContext, lipoLabel).getRelative(baseName);
      if (!gcdaFiles.contains(gcdaFile)) {
        // .gcda file not found
        return ImmutableList.of();
      }
    }

    final Artifact artifact = env.getDerivedArtifact(fdoPath.getRelative(gcdaFile), fdoRoot);
    env.registerAction(new FdoStubAction(ruleContext.getActionOwner(), artifact));

    return ImmutableList.of(artifact);
  }


  private PathFragment getAutoProfilePath() {
    return fdoRootExecPath.getRelative(getAutoProfileRootRelativePath());
  }

  private PathFragment getAutoProfileRootRelativePath() {
    return new PathFragment(fdoProfile.getBaseName());
  }


  private PathFragment getLLVMProfilePath() {
    return fdoRootExecPath.getRelative(getLLVMProfileRootRelativePath());
  }

  private PathFragment getLLVMProfileRootRelativePath() {
    return new PathFragment(fdoProfile.getBaseName());
  }

  /**
   * Returns whether LIPO is enabled.
   */
  @ThreadSafe
  public boolean isLipoEnabled() {
    return fdoProfile != null && lipoMode != LipoMode.OFF;
  }

  /**
   * Returns whether AutoFDO is enabled.
   */
  @ThreadSafe
  public boolean isAutoFdoEnabled() {
    return useAutoFdo;
  }

  /**
   * Adds the FDO profile output path to the variable builder.
   * If FDO is disabled, no build variable is added.
   */
  @ThreadSafe
  public void getLinkOptions(FeatureConfiguration featureConfiguration,
      CcToolchainFeatures.Variables.Builder buildVariables
      ) {
    if (featureConfiguration.isEnabled(CppRuleClasses.FDO_INSTRUMENT)) {
      buildVariables.addVariable("fdo_instrument_path",
          fdoInstrument.getPathString());
    }
  }

  /**
   * Returns the path of the FDO output tree (relative to the execution root)
   * containing the .gcda profile files, or null if FDO is not enabled.
   */
  @VisibleForTesting
  public PathFragment getFdoOptimizeDir() {
    return fdoRootExecPath;
  }

  /**
   * Returns the path of the FDO zip containing the .gcda profile files, or null
   * if FDO is not enabled.
   */
  @VisibleForTesting
  public Path getFdoOptimizeProfile() {
    return fdoProfile;
  }

  /**
   * Returns the path fragment of the instrumentation output dir for gcc when
   * FDO is enabled, or null if FDO is not enabled.
   */
  @ThreadSafe
  public PathFragment getFdoInstrument() {
    return fdoInstrument;
  }

  @VisibleForTesting
  public void setGcdaFilesForTesting(ImmutableSet<PathFragment> gcdaFiles) {
    this.gcdaFiles = gcdaFiles;
  }

  /**
   * An exception indicating an issue with FDO coverage files.
   */
  public static final class FdoException extends Exception {
    FdoException(String message) {
      super(message);
    }
  }
}
