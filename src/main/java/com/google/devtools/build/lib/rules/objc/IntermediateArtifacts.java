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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.cpp.CppModuleMap;
import com.google.devtools.build.lib.rules.cpp.CppModuleMap.UmbrellaHeaderStrategy;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Factory class for generating artifacts which are used as intermediate output.
 */
// TODO(bazel-team): This should really be named DerivedArtifacts as it contains methods for
// final as well as intermediate artifacts.
public final class IntermediateArtifacts {
  static final String LINKMAP_SUFFIX = ".linkmap";

  private final RuleContext ruleContext;
  private final BuildConfiguration buildConfiguration;
  private final String archiveFileNameSuffix;
  private final String outputPrefix;
  private final UmbrellaHeaderStrategy umbrellaHeaderStrategy;

  IntermediateArtifacts(RuleContext ruleContext, String archiveFileNameSuffix,
      String outputPrefix) {
    this(ruleContext, archiveFileNameSuffix, outputPrefix, ruleContext.getConfiguration(),
        UmbrellaHeaderStrategy.DO_NOT_GENERATE);
  }

  IntermediateArtifacts(RuleContext ruleContext, String archiveFileNameSuffix) {
    this(ruleContext, archiveFileNameSuffix, "", ruleContext.getConfiguration(),
        UmbrellaHeaderStrategy.DO_NOT_GENERATE);
  }

  IntermediateArtifacts(RuleContext ruleContext, String archiveFileNameSuffix,
      UmbrellaHeaderStrategy umbrellaHeaderStrategy) {
    this(ruleContext, archiveFileNameSuffix, "", ruleContext.getConfiguration(),
        umbrellaHeaderStrategy);
  }

  IntermediateArtifacts(RuleContext ruleContext, String archiveFileNameSuffix, String outputPrefix,
      BuildConfiguration buildConfiguration) {
    this(ruleContext, archiveFileNameSuffix, outputPrefix, buildConfiguration,
        UmbrellaHeaderStrategy.DO_NOT_GENERATE);
  }

  IntermediateArtifacts(RuleContext ruleContext, String archiveFileNameSuffix, String outputPrefix,
      BuildConfiguration buildConfiguration, UmbrellaHeaderStrategy umbrellaHeaderStrategy) {
    this.ruleContext = ruleContext;
    this.buildConfiguration = buildConfiguration;
    this.archiveFileNameSuffix = Preconditions.checkNotNull(archiveFileNameSuffix);
    this.outputPrefix = Preconditions.checkNotNull(outputPrefix);
    this.umbrellaHeaderStrategy = umbrellaHeaderStrategy;
  }

  /**
   * Returns a derived artifact in the bin directory obtained by appending some extension to the
   * main label name; the result artifact is placed in a unique "entitlements" directory.
   * For example, if this artifact is for a target Foo with extension ".extension", the result
   * artifact will be located at {target_base_path}/entitlements/Foo.extension.
   */
  public Artifact appendExtensionForEntitlementArtifact(String extension) {
    PathFragment entitlementsDirectory = ruleContext.getUniqueDirectory("entitlements");
    Artifact artifact =
        ruleContext.getDerivedArtifact(
            entitlementsDirectory.replaceName(
                addOutputPrefix(entitlementsDirectory.getBaseName(), extension)),
            buildConfiguration.getBinDirectory(ruleContext.getRule().getRepository()));
    return artifact;
  }

  /**
   * Returns the archive file name suffix.
   */
  public String archiveFileNameSuffix() {
    return archiveFileNameSuffix;
  }

  /**
   * Returns the location of this target's generated entitlements file.
   */
  public Artifact entitlements() {
    return appendExtensionForEntitlementArtifact(".entitlements");
  }

  /**
   * Returns the location of this target's extension plist which contains entries required by all
   * watch extensions (for final merging into the bundle plist).
   */
  public Artifact watchExtensionAutomaticPlist() {
    return ruleContext.getRelatedArtifact(
        ruleContext.getUniqueDirectory("plists"), "-automatic-watchExtensionInfo.plist");
  }

  /**
   * Returns a derived artifact in the bin directory obtained by appending some extension to the end
   * of the given {@link PathFragment}.
   */
  private Artifact appendExtension(PathFragment original, String extension) {
    return scopedArtifact(FileSystemUtils.appendExtension(original,
        addOutputPrefix("", extension)));
  }

  /**
   * Returns a derived artifact in the bin directory obtained by appending some extension to the end
   * of the {@link PathFragment} corresponding to the owner {@link Label}.
   */
  private Artifact appendExtension(String extension) {
    PathFragment name = PathFragment.create(ruleContext.getLabel().getName());
    return scopedArtifact(name.replaceName(addOutputPrefix(name.getBaseName(), extension)));
  }

  /**
   * Returns a derived artifact in the genfiles directory obtained by appending some extension to
   * the end of the {@link PathFragment} corresponding to the owner {@link Label}.
   */
  private Artifact appendExtensionInGenfiles(String extension) {
    PathFragment name = PathFragment.create(ruleContext.getLabel().getName());
    return scopedArtifact(
        name.replaceName(addOutputPrefix(name.getBaseName(), extension)), /* inGenfiles = */ true);
  }

  /**
   * The output of using {@code actoolzip} to run {@code actool} for a given bundle which is
   * merged under the {@code .app} or {@code .bundle} directory root.
   */
  public Artifact actoolzipOutput() {
    return appendExtension(".actool.zip");
  }

  /**
   * Output of the partial infoplist generated by {@code actool} when given the
   * {@code --output-partial-info-plist [path]} flag.
   */
  public Artifact actoolPartialInfoplist() {
    return appendExtension(".actool-PartialInfo.plist");
  }

  /**
   * The Info.plist file for a bundle which is comprised of more than one originating plist file.
   * This is not needed for a bundle which has no source Info.plist files, or only one Info.plist
   * file, since no merging occurs in that case.
   */
  public Artifact mergedInfoplist() {
    return appendExtension("-MergedInfo.plist");
  }

  /**
   * The .objlist file, which contains a list of paths of object files to archive and is read by
   * clang (via -filelist flag) in the link action (for binary creation).
   */
  public Artifact linkerObjList() {
    return appendExtension("-linker.objlist");
  }

  /**
   * The .objlist file, which contains a list of paths of object files to archive  and is read by
   * libtool (via -filelist flag) in the archive action.
   */
  public Artifact archiveObjList() {
    return appendExtension("-archive.objlist");
  }

  /**
   * The artifact which is the binary (or library) which is comprised of one or more .a files linked
   * together. Compared to the artifact returned by {@link #unstrippedSingleArchitectureBinary},
   * this artifact is stripped of symbol table when --compilation_mode=opt and
   * --objc_enable_binary_stripping are specified.
   */
  public Artifact strippedSingleArchitectureBinary() {
    return appendExtension("_bin");
  }

  /**
   * The artifact which is the fully-linked static library comprised of statically linking compiled
   * sources and dependencies together.
   */
  public Artifact strippedSingleArchitectureLibrary() {
    return appendExtension("-fl.a");
  }

  /**
   * The artifact which is the binary (or library) which is comprised of one or more .a files linked
   * together. It also contains full debug symbol information, compared to the artifact returned
   * by {@link #strippedSingleArchitectureBinary}. This artifact will serve as input for the symbol
   * strip action and is only created when --compilation_mode=opt and
   * --objc_enable_binary_stripping are specified.
   */
  public Artifact unstrippedSingleArchitectureBinary() {
    return appendExtension("_bin_unstripped");
  }

  /**
   * Lipo binary generated by combining one or more linked binaries. This binary is the one included
   * in generated bundles and invoked as entry point to the application.
   */
  public Artifact combinedArchitectureBinary() {
    return appendExtension("_lipobin");
  }

  /**
   * Lipo archive generated by combining one or more linked archives.
   */
  public Artifact combinedArchitectureArchive() {
    return appendExtension("_lipo.a");
  }

  /**
   * Lipo'ed dynamic library generated by combining one or more single-architecture linked dynamic
   * libraries.
   */
  public Artifact combinedArchitectureDylib() {
    return appendExtension("_lipo.dylib");
  }

  private Artifact scopedArtifact(PathFragment scopeRelative, boolean inGenfiles) {
    ArtifactRoot root =
        inGenfiles
            ? buildConfiguration.getGenfilesDirectory(ruleContext.getRule().getRepository())
            : buildConfiguration.getBinDirectory(ruleContext.getRule().getRepository());

    // The path of this artifact will be RULE_PACKAGE/SCOPERELATIVE
    return ruleContext.getPackageRelativeArtifact(scopeRelative, root);
  }

  private Artifact scopedArtifact(PathFragment scopeRelative) {
    return scopedArtifact(scopeRelative, /* inGenfiles = */ false);
  }

  /**
   * The {@code .a} file which contains all the compiled sources for a rule.
   */
  public Artifact archive() {
    // The path will be {RULE_PACKAGE}/lib{RULEBASENAME}{SUFFIX}.a
    String basename = PathFragment.create(ruleContext.getLabel().getName()).getBaseName();
    return scopedArtifact(PathFragment.create(String.format(
        "lib%s%s.a", basename, archiveFileNameSuffix)));
  }

  /** The artifact for the .headers file output by the header thinning action for this source. */
  public Artifact headersListFile(Artifact objectFile) {
    return ruleContext.getRelatedArtifact(objectFile.getRootRelativePath(), ".headers_list");
  }

  /**
   * The artifact which contains the zipped-up results of compiling the storyboard. This is merged
   * into the final bundle under the {@code .app} or {@code .bundle} directory root.
   */
  public Artifact compiledStoryboardZip(Artifact input) {
    return appendExtension("/" + BundleableFile.flatBundlePath(input.getExecPath()) + ".zip");
  }

  /**
   * Returns the artifact which is the output of building an entire xcdatamodel[d] made of artifacts
   * specified by a single rule.
   *
   * @param containerDir the containing *.xcdatamodeld or *.xcdatamodel directory
   * @return the artifact for the zipped up compilation results.
   */
  public Artifact compiledMomZipArtifact(PathFragment containerDir) {
    return appendExtension(
        "/" + FileSystemUtils.replaceExtension(containerDir, ".zip").getBaseName());
  }

  /**
   * Returns the compiled (i.e. converted to binary plist format) artifact corresponding to the
   * given {@code .strings} file.
   */
  public Artifact convertedStringsFile(Artifact originalFile) {
    return appendExtension(originalFile.getExecPath(), ".binary");
  }

  /**
   * Returns the artifact corresponding to the zipped-up compiled form of the given {@code .xib}
   * file.
   */
  public Artifact compiledXibFileZip(Artifact originalFile) {
    return appendExtension(
        "/" + FileSystemUtils.replaceExtension(originalFile.getExecPath(), ".nib.zip"));
  }

  /**
   * Returns the artifact which is the output of running swift-stdlib-tool and copying resulting
   * dylibs.
   */
  public Artifact swiftFrameworksFileZip() {
    return appendExtension(".swiftstdlib.zip");
  }

  /**
   * Same as {@link #swiftFrameworksFileZip()} but used to put Swift dylibs at a different location
   * in SwiftSupport directory at the top of the IPA.
   */
  public Artifact swiftSupportZip() {
    return appendExtension(".swiftsupport.zip");
  }

  /**
   * Debug symbol file generated for a stripped linked binary.
   *
   * <p>The name of the debug symbol file matches that of stripped binary plus that of the debug
   * symbol file extension (.dwarf), so we must know if the binary has been stripped
   * or not as that will modify its name.
   */
  public Artifact dsymSymbolForStrippedBinary() {
    return dsymSymbol("bin");
  }

  /**
   * Debug symbol file generated for an unstripped linked binary.
   *
   * <p>The name of the debug symbol file matches that of unstripped binary plus that of the debug
   * symbol file extension (.dwarf).
   */
  public Artifact dsymSymbolForUnstrippedBinary() {
    return dsymSymbol("bin_unstripped");
  }

  /**
   * Debug symbol file generated for a linked binary, for a specific architecture.
   */
  private Artifact dsymSymbol(String suffix) {
    return appendExtension(String.format("_%s.dwarf", suffix));
  }

  /** Bitcode symbol map generated for a linked binary, for a specific architecture. */
  public Artifact bitcodeSymbolMap() {
    return appendExtension(".bcsymbolmap");
  }

  /** Representation for a specific architecture. */
  private Artifact architectureRepresentation(String arch, String suffix) {
    return appendExtension(String.format("_%s%s", arch, suffix));
  }

  /**
   * Linkmap representation
   */
  public Artifact linkmap() {
    return appendExtension(LINKMAP_SUFFIX);
  }

  /**
   * Linkmap representation for a specific architecture.
   */
  public Artifact linkmap(String arch) {
    return architectureRepresentation(arch, LINKMAP_SUFFIX);
  }

  /**
   * {@link CppModuleMap} that provides the clang module map for this target.
   */
  public CppModuleMap moduleMap() {
    String moduleName;
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("module_name")) {
      moduleName = ruleContext.attributes().get("module_name", Type.STRING);
    } else {
      moduleName =
          ruleContext
              .getLabel()
              .toString()
              .replace("//", "")
              .replace("@", "")
              .replace("-", "_")
              .replace("/", "_")
              .replace(":", "_");
    }
    Optional<Artifact> customModuleMap = CompilationSupport.getCustomModuleMap(ruleContext);
    if (customModuleMap.isPresent()) {
      return new CppModuleMap(customModuleMap.get(), moduleName);
    } else if (umbrellaHeaderStrategy == UmbrellaHeaderStrategy.GENERATE) {
      // To get Swift to pick up module maps, we need to name them "module.modulemap" and have their
      // parent directory in the module map search paths.
      return new CppModuleMap(
          appendExtensionInGenfiles(".modulemaps/module.modulemap"),
          appendExtensionInGenfiles(".modulemaps/umbrella.h"),
          moduleName);
    } else {
      return new CppModuleMap(
          appendExtensionInGenfiles(".modulemaps/module.modulemap"), moduleName);
    }
  }

  /**
   * Returns a static library archive with dead code/objects removed by J2ObjC dead code removal,
   * given the original unpruned static library containing J2ObjC-translated code.
   */
  public Artifact j2objcPrunedArchive(Artifact unprunedArchive) {
    PathFragment prunedSourceArtifactPath = FileSystemUtils.appendWithoutExtension(
        unprunedArchive.getRootRelativePath(), "_pruned");
    return ruleContext.getUniqueDirectoryArtifact(
        "_j2objc_pruned",
        prunedSourceArtifactPath,
        buildConfiguration.getBinDirectory(ruleContext.getRule().getRepository()));
  }

  /**
   * Returns the location of this target's merged but not post-processed or signed IPA.
   */
  public Artifact unprocessedIpa() {
    return appendExtension(".unprocessed.ipa");
  }

  /**
   * Returns artifact name prefixed with an output prefix if specified.
   */
  private String addOutputPrefix(String baseName, String artifactName) {
    if (!outputPrefix.isEmpty()) {
      return String.format("%s-%s%s", baseName, outputPrefix, artifactName);
    }
    return String.format("%s%s", baseName, artifactName);
  }

}
