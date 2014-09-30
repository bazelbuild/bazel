// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadHostile;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.skyframe.BuildVariableValue;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.ZipFileSystem;
import com.google.devtools.build.lib.view.AnalysisEnvironment;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.LipoMode;
import com.google.devtools.build.skyframe.SkyFunction;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentMap;
import java.util.regex.Pattern;
import java.util.zip.ZipException;

/**
 * Support class for FDO (feedback directed optimization) and LIPO (lightweight inter-procedural
 * optimization).
 *
 * <p>There is a 1:1 relationship between {@link BuildConfiguration} objects and {@code FdoSupport}
 * objects. The FDO support of a build configuration can be retrieved using {@link
 * CppConfiguration#getFdoSupport()}.
 *
 * <p>With respect to thread-safety, the {@link #prepareToBuild} method is not thread-safe, and must
 * not be called concurrently with other methods on this class.
 */
public class FdoSupport implements Serializable {

  /**
   * Path within profile data .zip files that is considered the root of the
   * profile information directory tree.
   */
  private static final PathFragment ZIP_ROOT = new PathFragment("/");

  /**
   * Returns true if the give fdoFile represents an AutoFdo profile.
   */
  public static final boolean isAutoFdo(String fdoFile) {
    return CppFileTypes.GCC_AUTO_PROFILE.matches(fdoFile);
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
  private ImmutableMultimap<PathFragment, Artifact> imports;

  /**
   * Registered scannables for auxiliary source files.
   */
  private final ConcurrentMap<PathFragment, IncludeScannable> scannables = Maps.newConcurrentMap();

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
        : fdoRoot.getExecPath().getRelative(new PathFragment("_fdo").getChild(
            FileSystemUtils.removeExtension(fdoProfile.getBaseName())));
    this.fdoPath = fdoProfile == null
        ? null
        : new PathFragment("_fdo").getChild(
            FileSystemUtils.removeExtension(fdoProfile.getBaseName()));
    this.lipoMode = lipoMode;
    this.useAutoFdo = fdoProfile != null && isAutoFdo(fdoProfile.getBaseName());
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
        BuildVariableValue.dependOnBuildId(env);
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
  public void prepareToBuild(Path execRoot, PathFragment genfilesPath,
      ArtifactFactory artifactDeserializer) throws IOException, FdoException {
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
          imports = readAutoFdoImports(artifactDeserializer, fdoImports, genfilesPath);
        }
        FileSystemUtils.ensureSymbolicLink(
            execRoot.getRelative(getAutoProfilePath()), fdoProfile);
      } else {
        Path zipFilePath = new ZipFileSystem(fdoProfile).getRootDirectory();
        if (!zipFilePath.getRelative("blaze-out").isDirectory()) {
          throw new ZipException("FDO zip files must be zipped directly above 'blaze-out' " +
                                 "for the compiler to find the profile");
        }
        ImmutableSet.Builder<PathFragment> gcdaFilesBuilder = ImmutableSet.builder();
        ImmutableMultimap.Builder<PathFragment, Artifact> importsBuilder =
            ImmutableMultimap.builder();
        extractFdoZip(artifactDeserializer, zipFilePath, fdoDirPath,
            gcdaFilesBuilder, importsBuilder);
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
  private void extractFdoZip(ArtifactFactory artifactFactory, Path sourceDir,
      Path targetDir, ImmutableSet.Builder<PathFragment> gcdaFilesBuilder,
      ImmutableMultimap.Builder<PathFragment, Artifact> importsBuilder)
          throws IOException, FdoException {
    for (Path sourceFile : sourceDir.getDirectoryEntries()) {
      Path targetFile = targetDir.getRelative(sourceFile.getBaseName());
      if (sourceFile.isDirectory()) {
        targetFile.createDirectory();
        extractFdoZip(artifactFactory, sourceFile, targetFile, gcdaFilesBuilder, importsBuilder);
      } else {
        if (CppFileTypes.COVERAGE_DATA.matches(sourceFile)) {
          FileSystemUtils.copyFile(sourceFile, targetFile);
          gcdaFilesBuilder.add(
              sourceFile.relativeTo(sourceFile.getFileSystem().getRootDirectory()));
        } else if (CppFileTypes.COVERAGE_DATA_IMPORTS.matches(sourceFile)) {
          readCoverageImports(artifactFactory, sourceFile, importsBuilder);
        } else {
            throw new FdoException("FDO ZIP file contained a file of unknown type: "
                + sourceFile);
        }
      }
    }
  }

  /**
   * Reads a .gcda.imports file and stores the imports information.
   *
   * @throws FdoException if an auxiliary LIPO input was not found
   */
  private void readCoverageImports(ArtifactFactory artifactFactory, Path importsFile,
      ImmutableMultimap.Builder<PathFragment, Artifact> importsBuilder)
          throws IOException, FdoException {
    PathFragment key = importsFile.asFragment().relativeTo(ZIP_ROOT);
    String baseName = key.getBaseName();
    String ext = Iterables.getOnlyElement(CppFileTypes.COVERAGE_DATA_IMPORTS.getExtensions());
    key = key.replaceName(baseName.substring(0, baseName.length() - ext.length()));

    for (String line : FileSystemUtils.iterateLinesAsLatin1(importsFile)) {
      if (!line.isEmpty()) {
        // We can't yet fully check the validity of a line. this is done later
        // when we actually parse the contained paths.
        Artifact artifact = artifactFactory.deserializeArtifact(
            new PathFragment(line));
        if (artifact == null) {
          throw new FdoException("Auxiliary LIPO input not found: " + line);
        }

        importsBuilder.put(key, artifact);
      }
    }
  }

  /**
   * Reads a .afdo.imports file and stores the imports information.
   */
  private ImmutableMultimap<PathFragment, Artifact> readAutoFdoImports(
      ArtifactFactory artifactFactory, Path importsFile, PathFragment genFilePath)
          throws IOException, FdoException {
    ImmutableMultimap.Builder<PathFragment, Artifact> importBuilder = ImmutableMultimap.builder();
    for (String line : FileSystemUtils.iterateLinesAsLatin1(importsFile)) {
      if (!line.isEmpty()) {
        PathFragment key = new PathFragment(line.substring(0, line.indexOf(':')));
        if (key.startsWith(genFilePath)) {
          key = key.relativeTo(genFilePath);
        }
        key = FileSystemUtils.replaceSegments(key, "PROTECTED", "_protected", true);
        for (String auxFile : line.substring(line.indexOf(':') + 1).split(" ")) {
          if (auxFile.length() == 0) {
            continue;
          }
          Artifact artifact = artifactFactory.deserializeArtifact(
              new PathFragment(auxFile));
          if (artifact == null) {
            throw new FdoException("Auxiliary LIPO input not found: " + auxFile);
          }
          importBuilder.put(key, artifact);
        }
      }
    }
    return importBuilder.build();
  }

  /**
   * Returns the imports from the .afdo.imports file of a source file.
   *
   * @param sourceName the source file
   */
  private Iterable<Artifact> getAutoFdoImports(PathFragment sourceName) {
    Preconditions.checkState(isLipoEnabled());
    ImmutableCollection<Artifact> afdoImports = imports.get(sourceName);
    Preconditions.checkState(afdoImports != null,
        "AutoFDO import data missing for %s", sourceName);
    return afdoImports;
  }

  /**
   * Returns the imports from the .gcda.imports file of an object file.
   *
   * @param objDirectory the object directory of the object file's target
   * @param objectName the object file
   */
  private Iterable<Artifact> getImports(PathFragment objDirectory, PathFragment objectName) {
    Preconditions.checkState(isLipoEnabled());
    Preconditions.checkState(imports != null,
        "Tried to look up imports of uninitialized FDOSupport");
    PathFragment key = objDirectory.getRelative(FileSystemUtils.removeExtension(objectName));
    ImmutableCollection<Artifact> importsForObject = imports.get(key);
    Preconditions.checkState(importsForObject != null, "Import data missing for %s", key);
    return importsForObject;
  }

  /**
   * Configures a compile action builder by adding command line options and
   * auxiliary inputs according to the FDO configuration. This method does
   * nothing If FDO is disabled.
   */
  @ThreadSafe
  public void configureCompilation(CppCompileActionBuilder builder, RuleContext ruleContext,
      AnalysisEnvironment env, Label lipoLabel, PathFragment sourceName, final Pattern nocopts,
      boolean usePic, LipoContextProvider lipoInputProvider) {
    // It is a bug if this method is called with useLipo if lipo is disabled. However, it is legal
    // if is is called with !useLipo, even though lipo is enabled.
    Preconditions.checkArgument(lipoInputProvider == null || isLipoEnabled());

    // FDO is disabled -> do nothing.
    if ((fdoInstrument == null) && (fdoRoot == null)) {
      return;
    }

    List<String> fdoCopts = new ArrayList<>();
    // Instrumentation phase
    if (fdoInstrument != null) {
      fdoCopts.add("-fprofile-generate=" + fdoInstrument.getPathString());
      if (lipoMode != LipoMode.OFF) {
        fdoCopts.add("-fripa");
      }
    }

    // Optimization phase
    if (fdoRoot != null) {
      // Declare dependency on contents of zip file.
      if (env.getSkyframeEnv().valuesMissing()) {
        return;
      }
      Iterable<Artifact> auxiliaryInputs = getAuxiliaryInputs(
          ruleContext, env, lipoLabel, sourceName, usePic, lipoInputProvider);
      builder.addMandatoryInputs(auxiliaryInputs);
      if (!Iterables.isEmpty(auxiliaryInputs)) {
        if (useAutoFdo) {
          fdoCopts.add("-fauto-profile=" + getAutoProfilePath().getPathString());
        } else {
          fdoCopts.add("-fprofile-use=" + fdoRootExecPath);
        }
        fdoCopts.add("-fprofile-correction");
        if (lipoInputProvider != null) {
          fdoCopts.add("-fripa");
        }
      }
    }
    Iterable<String> filteredCopts = fdoCopts;
    if (nocopts != null) {
      // Filter fdoCopts with nocopts if they exist.
      filteredCopts = Iterables.filter(fdoCopts, new Predicate<String>() {
        @Override
        public boolean apply(String copt) {
          return !nocopts.matcher(copt).matches();
        }
      });
    }
    builder.addCopts(0, filteredCopts);
  }

  /**
   * Returns the auxiliary files that need to be added to the {@link CppCompileAction}.
   */
  private Iterable<Artifact> getAuxiliaryInputs(
      RuleContext ruleContext, AnalysisEnvironment env, Label lipoLabel, PathFragment sourceName,
      boolean usePic, LipoContextProvider lipoContextProvider) {
    // If --fdo_optimize was not specified, we don't have any additional inputs.
    if (fdoProfile == null) {
      return ImmutableSet.of();
    } else if (useAutoFdo) {
      ImmutableSet.Builder<Artifact> auxiliaryInputs = ImmutableSet.builder();

      Artifact artifact = env.getDerivedArtifact(
          fdoPath.getRelative(getAutoProfileRootRelativePath()), fdoRoot);
      env.registerAction(new FdoStubAction(ruleContext.getActionOwner(), artifact));
      auxiliaryInputs.add(artifact);
      if (lipoContextProvider != null) {
        for (Artifact importedFile : getAutoFdoImports(sourceName)) {
          auxiliaryInputs.add(importedFile);
        }
      }
      return auxiliaryInputs.build();
    } else {
      ImmutableSet.Builder<Artifact> auxiliaryInputs = ImmutableSet.builder();

      PathFragment objectName =
          FileSystemUtils.replaceExtension(sourceName, usePic ? ".pic.o" : ".o");

      auxiliaryInputs.addAll(
          getGcdaArtifactsForObjectFileName(ruleContext, env, objectName, lipoLabel));

      if (lipoContextProvider != null) {
        for (Artifact importedFile : getImports(
            getNonLipoObjDir(ruleContext, lipoLabel), objectName)) {
          if (CppFileTypes.COVERAGE_DATA.matches(importedFile.getFilename())) {
            try {
              auxiliaryInputs.addAll(getGcdaArtifactsForGcdaPath(
                  ruleContext, env, importedFile.getExecPath(), lipoContextProvider));
            } catch (FileNotFoundException e) {
              ruleContext.ruleError(e.getMessage());
            } catch (NoSuchTargetException e) {
              ruleContext.ruleError(e.getMessage());
            }
          } else {
            auxiliaryInputs.add(importedFile);
          }
        }
      }

      return auxiliaryInputs.build();
    }
  }

  /**
   * Returns a list of .gcda file artifacts for a .gcda path from the
   * .gcda.imports file.
   *
   * <p>The resulting set contains either one or two artifacts (the file itself
   * and a symlink to it).
   *
   * @throws FileNotFoundException if no .gcda file could be found
   * @throws NoSuchTargetException if the target for {@code gcdaPath} could not
   *         be determined
   */
  private ImmutableList<Artifact> getGcdaArtifactsForGcdaPath(RuleContext ruleContext,
      AnalysisEnvironment env, PathFragment gcdaPath, LipoContextProvider lipoContextProvider)
      throws FileNotFoundException, NoSuchTargetException {
    Map.Entry<PathFragment, Label> entry =
        lipoContextProvider.getPathsToLabels().floorEntry(gcdaPath);

    if (entry == null || !gcdaPath.startsWith(entry.getKey())) {
      throw new NoSuchTargetException("Target not found for .gcda file: " + gcdaPath);
    }

    // The label of the target which owns the .gcda file
    Label lipoLabel = entry.getValue();

    PathFragment sourceName = gcdaPath.relativeTo(getNonLipoObjDir(ruleContext, lipoLabel));
    ImmutableList<Artifact> result =
        getGcdaArtifactsForObjectFileName(ruleContext, env, sourceName, lipoLabel);

    if (result.isEmpty()) {
      // Here we know that there should be .gcda file. If the returned set is
      // empty something went wrong.
      throw new FileNotFoundException("Auxiliary LIPO input not found: " + gcdaPath);
    }

    return result;
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
   * Returns an immutable list of command line arguments to add to the linker
   * command line. If FDO is disabled, and empty list is returned.
   */
  @ThreadSafe
  public ImmutableList<String> getLinkOptions() {
    return fdoInstrument != null
        ? ImmutableList.of("-fprofile-generate=" + fdoInstrument.getPathString())
        : ImmutableList.<String>of();
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

  /**
   * Registers a scannable that needs to be scanned when using a source file as
   * auxiliary source.
   *
   * @param source exec path of the source file
   * @param scannable the scannable to scan
   */
  @ThreadSafe
  public void registerScannable(PathFragment source, IncludeScannable scannable) {
    scannables.put(source, scannable);
  }

  /**
   * Gets the scannable for a source file. May return null if no scannable has
   * been registered with
   * {@link #registerScannable(PathFragment, IncludeScannable)}.
   *
   * @param source exec path of the source file
   */
  @ThreadSafe
  public IncludeScannable getScannable(PathFragment source) {
    return scannables.get(source);
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
