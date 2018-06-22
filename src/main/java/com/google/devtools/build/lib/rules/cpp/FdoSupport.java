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

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.io.ByteSource;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Enumeration;
import java.util.zip.ZipEntry;
import java.util.zip.ZipException;
import java.util.zip.ZipFile;

/**
 * Support class for FDO (feedback directed optimization).
 *
 * <p>Here follows a quick run-down of how FDO builds work (for non-FDO builds, none of this
 * applies):
 *
 * <p>{@link FdoSupport#create} is called from {@link FdoSupportFunction} (a {@link SkyFunction}),
 * which is requested from Skyframe by the {@code cc_toolchain} rule. It extracts the FDO .zip (in
 * case we work with an explicitly generated FDO profile file) or analyzes the .afdo.imports file
 * next to the .afdo file (if AutoFDO is in effect).
 *
 * <p>.afdo.imports files contain one import a line. A line is two paths separated by a colon, with
 * functions in the second path being referenced by functions in the first path. These are then put
 * into the imports map. If we do AutoFDO, we don't handle individual .gcda files, so gcdaFiles will
 * be empty.
 *
 * <p>Regular .fdo zip files contain .gcda files (which are added to gcdaFiles) and .gcda.imports
 * files. There is one .gcda.imports file for every source file and it contains one path in every
 * line, which can either be a path to a source file that contains a function referenced by the
 * original source file or the .gcda file for such a referenced file. They both are added to the
 * imports map.
 *
 * <p>For each C++ compile action in the target configuration, {@link #configureCompilation} is
 * called, which adds command line options and input files required for the build. There are 2
 * cases:
 *
 * <ul>
 *   <li>If we do AutoFDO, the .afdo file and the source files containing the functions imported by
 *       the original source file (as determined from the inputs map) are added.
 *   <li>If we do FDO, the .gcda file corresponding to the source file is added.
 * </ul>
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
   * The {@code .gcda} files that have been extracted from the ZIP file,
   * relative to the root of the ZIP file.
   */
  private final ImmutableSet<PathFragment> gcdaFiles;

  /**
   * Multimap from .gcda file base names to auxiliary input files.
   *
   * <p>The keys of the multimap are the exec root relative paths of .gcda files
   * with the extension removed. The values are the lines from the accompanying
   * .gcda.imports file.
   *
   * <p>The contents of the multimap are copied verbatim from the .gcda.imports
   * files and not yet checked for validity.
   */
  private final ImmutableMultimap<PathFragment, PathFragment> imports;

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
      Path fdoProfile,
      FdoZipContents fdoZipContents) {
    this.fdoInstrument = fdoInstrument;
    this.fdoProfile = fdoProfile;
    this.fdoRoot = fdoRoot;
    this.fdoRootExecPath = fdoRootExecPath;
    this.fdoPath = fdoProfile == null
        ? null
        : FileSystemUtils.removeExtension(PathFragment.create("_fdo").getChild(
            fdoProfile.getBaseName()));
    this.fdoMode = fdoMode;
    if (fdoZipContents != null) {
      this.gcdaFiles = fdoZipContents.gcdaFiles;
      this.imports = fdoZipContents.imports;
    } else {
      this.gcdaFiles = null;
      this.imports = null;
    }
  }

  public Path getFdoProfile() {
    return fdoProfile;
  }

  @VisibleForSerialization
  // This method only exists for serialization.
  FdoZipContents getFdoZipContents() {
      return gcdaFiles == null ? null : new FdoZipContents(gcdaFiles, imports);
  }

  /** Creates an initialized {@link FdoSupport} instance. */
  static FdoSupport create(
      SkyFunction.Environment env,
      String fdoInstrument,
      Path fdoProfile,
      Path execRoot,
      String productName,
      FdoMode fdoMode)
      throws IOException, FdoException, InterruptedException {

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

    if (fdoMode == FdoMode.LLVM_FDO) {
      return new FdoSupport(fdoMode, fdoRoot, fdoRootExecPath, fdoInstrument, fdoProfile, null);
    }

    FdoZipContents fdoZipContents =
        extractFdoZip(fdoMode, execRoot, fdoProfile, fdoRootExecPath, productName);
    return new FdoSupport(
        fdoMode, fdoRoot, fdoRootExecPath, fdoInstrument, fdoProfile, fdoZipContents);
  }

  @Immutable
  @AutoCodec
  static class FdoZipContents {
    public static final ObjectCodec<FdoZipContents> CODEC =
        new FdoSupport_FdoZipContents_AutoCodec();

    private final ImmutableSet<PathFragment> gcdaFiles;
    private final ImmutableMultimap<PathFragment, PathFragment> imports;

    @VisibleForSerialization
    @AutoCodec.Instantiator
    FdoZipContents(ImmutableSet<PathFragment> gcdaFiles,
        ImmutableMultimap<PathFragment, PathFragment> imports) {
      this.gcdaFiles = gcdaFiles;
      this.imports = imports;
    }
  }

  /**
   * Extracts the FDO zip file and collects data from it that's needed during analysis.
   *
   * <p>When an {@code --fdo_optimize} compile is requested, unpacks the given FDO zip file
   * into a clean working directory under execRoot.
   *
   * @throws FdoException if the FDO ZIP contains a file of unknown type
   */
  private static FdoZipContents extractFdoZip(
      FdoMode fdoMode,
      Path execRoot,
      Path fdoProfile,
      PathFragment fdoRootExecPath,
      String productName)
      throws IOException, FdoException {
    // The execRoot != null case is only there for testing. We cannot provide a real ZIP file in
    // tests because ZipFileSystem does not work with a ZIP on an in-memory file system.
    // IMPORTANT: Keep in sync with #declareSkyframeDependencies to avoid incrementality issues.
    ImmutableSet<PathFragment> gcdaFiles = ImmutableSet.of();
    ImmutableMultimap<PathFragment, PathFragment> imports = ImmutableMultimap.of();

    if (fdoProfile != null && execRoot != null) {
      Path fdoDirPath = execRoot.getRelative(fdoRootExecPath);

      FileSystemUtils.deleteTreesBelow(fdoDirPath);
      FileSystemUtils.createDirectoryAndParents(fdoDirPath);

      if (fdoMode == FdoMode.AUTO_FDO || fdoMode == FdoMode.XBINARY_FDO) {
        FileSystemUtils.ensureSymbolicLink(
            execRoot.getRelative(getAutoProfilePath(fdoProfile, fdoRootExecPath)), fdoProfile);
      } else {
        // Path objects referring to inside the zip file are only valid within this try block.
        // FdoZipContents doesn't reference any of them, so we are fine.
        if (!fdoProfile.exists()) {
          throw new FileNotFoundException(String.format("File '%s' does not exist", fdoProfile));
        }
        File zipIoFile = fdoProfile.getPathFile();
        File tempFile = null;
        try {
          // If it doesn't exist, we're dealing with an in-memory fs for testing
          // Copy the file and delete later
          if (!zipIoFile.exists()) {
            tempFile = File.createTempFile("bazel.test.", ".tmp");
            zipIoFile = tempFile;
            byte[] contents = FileSystemUtils.readContent(fdoProfile);
            try (OutputStream os = new FileOutputStream(tempFile)) {
              os.write(contents);
            }
          }
          try (ZipFile zipFile = new ZipFile(zipIoFile)) {
            String outputSymlinkName = productName + "-out";
            ImmutableSet.Builder<PathFragment> gcdaFilesBuilder = ImmutableSet.builder();
            ImmutableMultimap.Builder<PathFragment, PathFragment> importsBuilder =
                ImmutableMultimap.builder();
            extractFdoZipDirectory(
                zipFile, fdoDirPath, outputSymlinkName, gcdaFilesBuilder, importsBuilder);
            gcdaFiles = gcdaFilesBuilder.build();
            imports = importsBuilder.build();
          }
        } finally {
          if (tempFile != null) {
            tempFile.delete();
          }
        }
      }
    }

    return new FdoZipContents(gcdaFiles, imports);
  }

  /**
   * Recursively extracts a directory from the GCDA ZIP file into a target directory.
   *
   * <p>Imports files are not written to disk. Their content is directly added to an internal data
   * structure.
   *
   * <p>The files are written at $EXECROOT/blaze-fdo/_fdo/(base name of profile zip), and the {@code
   * _fdo} directory there is symlinked to from the exec root, so that the file are also available
   * at $EXECROOT/_fdo/..., which is their exec path. We need to jump through these hoops because
   * the FDO root 1. needs to be a source root, thus the exec path of its root is ".", 2. it must
   * not be equal to the exec root so that the artifact factory does not get confused, 3. the files
   * under it must be reachable by their exec path from the exec root.
   *
   * @throws IOException if any of the I/O operations failed
   * @throws FdoException if the FDO ZIP contains a file of unknown type
   */
  private static void extractFdoZipDirectory(
      ZipFile zipFile,
      Path targetDir,
      String outputSymlinkName,
      ImmutableSet.Builder<PathFragment> gcdaFilesBuilder,
      ImmutableMultimap.Builder<PathFragment, PathFragment> importsBuilder)
      throws IOException, FdoException {
    Enumeration<? extends ZipEntry> zipEntries = zipFile.entries();
    while (zipEntries.hasMoreElements()) {
      ZipEntry zipEntry = zipEntries.nextElement();
      if (zipEntry.isDirectory()) {
        continue;
      }
      String name = zipEntry.getName();
      if (!name.startsWith(outputSymlinkName)) {
        throw new ZipException(
            "FDO zip files must be zipped directly above '"
                + outputSymlinkName
                + "' for the compiler to find the profile");
      }
      Path targetFile = targetDir.getRelative(name);
      FileSystemUtils.createDirectoryAndParents(targetFile.getParentDirectory());
      if (CppFileTypes.COVERAGE_DATA.matches(name)) {
        try (OutputStream out = targetFile.getOutputStream()) {
          new ZipByteSource(zipFile, zipEntry).copyTo(out);
        }
        targetFile.setLastModifiedTime(zipEntry.getTime());
        targetFile.setWritable(false);
        targetFile.setExecutable(false);
        gcdaFilesBuilder.add(PathFragment.create(name));
      } else if (CppFileTypes.COVERAGE_DATA_IMPORTS.matches(name)) {
        readCoverageImports(zipFile, zipEntry, importsBuilder);
      }
    }
  }

  private static class ZipByteSource extends ByteSource {
    final ZipFile zipFile;
    final ZipEntry zipEntry;

    ZipByteSource(ZipFile zipFile, ZipEntry zipEntry) {
      this.zipFile = zipFile;
      this.zipEntry = zipEntry;
    }

    @Override
    public InputStream openStream() throws IOException {
      return zipFile.getInputStream(zipEntry);
    }
  }

  /**
   * Reads a .gcda.imports file and stores the imports information.
   *
   * @throws FdoException
   */
  private static void readCoverageImports(
      ZipFile zipFile,
      ZipEntry importsFile,
      ImmutableMultimap.Builder<PathFragment, PathFragment> importsBuilder)
      throws IOException, FdoException {
    PathFragment key = PathFragment.create(importsFile.getName());
    String baseName = key.getBaseName();
    String ext = Iterables.getOnlyElement(CppFileTypes.COVERAGE_DATA_IMPORTS.getExtensions());
    key = key.replaceName(baseName.substring(0, baseName.length() - ext.length()));
    for (String line :
        new ZipByteSource(zipFile, importsFile).asCharSource(ISO_8859_1).readLines()) {
      if (!line.isEmpty()) {
        // We can't yet fully check the validity of a line. this is done later
        // when we actually parse the contained paths.
        PathFragment execPath = PathFragment.create(line);
        if (execPath.isAbsolute()) {
          throw new FdoException(
              "Absolute paths not allowed in gcda imports file " + importsFile + ": " + execPath);
        }
        importsBuilder.put(key, PathFragment.create(line));
      }
    }
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
      PathFragment outputName,
      boolean usePic,
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
          getAuxiliaryInputs(ruleContext, outputName, usePic, fdoSupportProvider);
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
  private Iterable<Artifact> getAuxiliaryInputs(
      RuleContext ruleContext,
      PathFragment outputName,
      boolean usePic,
      FdoSupportProvider fdoSupportProvider) {
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
      PathFragment objectName =
          FileSystemUtils.appendExtension(outputName, usePic ? ".pic.o" : ".o");

      auxiliaryInputs.addAll(
          getGcdaArtifactsForObjectFileName(ruleContext, fdoSupportProvider, objectName));

      return auxiliaryInputs.build();
    }
  }

  private PathFragment getObjDir(RuleContext ruleContext) {
    return ruleContext
        .getConfiguration()
        .getBinFragment()
        .getRelative(CppHelper.getObjDirectory(ruleContext.getLabel()));
  }

  /**
   * Returns a list of .gcda file artifacts for an object file path.
   *
   * <p>The resulting set is either empty (because no .gcda file exists for the given object file)
   * or contains one or two artifacts (the file itself and a symlink to it).
   */
  private ImmutableList<Artifact> getGcdaArtifactsForObjectFileName(
      RuleContext ruleContext, FdoSupportProvider fdoSupportProvider, PathFragment objectFileName) {
    // We put the .gcda files relative to the location of the .o file in the instrumentation run.
    String gcdaExt = Iterables.getOnlyElement(CppFileTypes.COVERAGE_DATA.getExtensions());
    PathFragment baseName = FileSystemUtils.replaceExtension(objectFileName, gcdaExt);
    PathFragment gcdaFile = getObjDir(ruleContext).getRelative(baseName);

    if (!gcdaFiles.contains(gcdaFile)) {
      // If the object is a .pic.o file and .pic.gcda is not found, we should try finding .gcda too
      String picoExt = Iterables.getOnlyElement(CppFileTypes.PIC_OBJECT_FILE.getExtensions());
      baseName = FileSystemUtils.replaceExtension(objectFileName, gcdaExt, picoExt);
      if (baseName == null) {
        // Object file is not .pic.o
        return ImmutableList.of();
      }
      gcdaFile = getObjDir(ruleContext).getRelative(baseName);
      if (!gcdaFiles.contains(gcdaFile)) {
        // .gcda file not found
        return ImmutableList.of();
      }
    }

    return ImmutableList.of(fdoSupportProvider.getGcdaArtifacts().get(gcdaFile));
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
      return new FdoSupportProvider(this, profiles, null);
    }

    if (fdoMode == FdoMode.LLVM_FDO) {
      Preconditions.checkState(profiles != null && profiles.getProfileArtifact() != null);
      return new FdoSupportProvider(this, profiles, null);
    }

    Preconditions.checkState(fdoPath != null);
    PathFragment profileRootRelativePath = getAutoProfileRootRelativePath(fdoProfile);

    Artifact profileArtifact =
        ruleContext
            .getAnalysisEnvironment()
            .getDerivedArtifact(fdoPath.getRelative(profileRootRelativePath), fdoRoot);
    ruleContext.registerAction(new FdoStubAction(ruleContext.getActionOwner(), profileArtifact));
    Preconditions.checkState(fdoPath != null);
    ImmutableMap.Builder<PathFragment, Artifact> gcdaArtifacts = ImmutableMap.builder();
    for (PathFragment path : gcdaFiles) {
      Artifact gcdaArtifact = ruleContext.getAnalysisEnvironment().getDerivedArtifact(
          fdoPath.getRelative(path), fdoRoot);
      ruleContext.registerAction(new FdoStubAction(ruleContext.getActionOwner(), gcdaArtifact));
      gcdaArtifacts.put(path, gcdaArtifact);
    }

    return new FdoSupportProvider(
        this,
        new ProfileArtifacts(
            profileArtifact, profiles == null ? null : profiles.getPrefetchHintsArtifact()),
        gcdaArtifacts.build());
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
