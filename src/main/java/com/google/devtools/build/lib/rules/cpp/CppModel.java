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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.cpp.CcCompilationOutputs.Builder;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.Link.LinkStaticness;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.regex.Pattern;

import javax.annotation.Nullable;

/**
 * Representation of a C/C++ compilation. Its purpose is to share the code that creates compilation
 * actions between all classes that need to do so. It follows the builder pattern - load up the
 * necessary settings and then call {@link #createCcCompileActions}.
 *
 * <p>This class is not thread-safe, and it should only be used once for each set of source files,
 * i.e. calling {@link #createCcCompileActions} will throw an Exception if called twice.
 */
public final class CppModel {
  private final CppSemantics semantics;
  private final RuleContext ruleContext;
  private final BuildConfiguration configuration;
  private final CppConfiguration cppConfiguration;

  // compile model
  private CppCompilationContext context;
  private final List<Pair<Artifact, Label>> sourceFiles = new ArrayList<>();
  private final List<String> copts = new ArrayList<>();
  private final List<PathFragment> additionalIncludes = new ArrayList<>();
  @Nullable private Pattern nocopts;
  private boolean fake;
  private boolean maySaveTemps;
  private boolean onlySingleOutput;
  private CcCompilationOutputs compilationOutputs;
  private boolean enableLayeringCheck;
  private boolean compileHeaderModules;

  // link model
  private final List<String> linkopts = new ArrayList<>();
  private LinkTargetType linkType = LinkTargetType.STATIC_LIBRARY;
  private boolean neverLink;
  private boolean allowInterfaceSharedObjects;
  private boolean createDynamicLibrary = true;
  private PathFragment soImplFilename;
  private FeatureConfiguration featureConfiguration;

  public CppModel(RuleContext ruleContext, CppSemantics semantics) {
    this.ruleContext = ruleContext;
    this.semantics = semantics;
    configuration = ruleContext.getConfiguration();
    cppConfiguration = configuration.getFragment(CppConfiguration.class);
  }

  /**
   * If the cpp compilation is a fake, then it creates only a single compile action without PIC.
   * Defaults to false.
   */
  public CppModel setFake(boolean fake) {
    this.fake = fake;
    return this;
  }

  /**
   * If set, the CppModel only creates a single .o output that can be linked into a dynamic library,
   * i.e., it never generates both PIC and non-PIC outputs. Otherwise it creates outputs that can be
   * linked into both static binaries and dynamic libraries (if both require PIC or both require
   * non-PIC, then it still only creates a single output). Defaults to false.
   */
  public CppModel setOnlySingleOutput(boolean onlySingleOutput) {
    this.onlySingleOutput = onlySingleOutput;
    return this;
  }

  /**
   * If set, use compiler flags to enable compiler based layering checks.
   */
  public CppModel setEnableLayeringCheck(boolean enableLayeringCheck) {
    this.enableLayeringCheck = enableLayeringCheck;
    return this;
  }

  /**
   * If set, add actions that compile header modules to the build.
   * See http://clang.llvm.org/docs/Modules.html for more information.
   */
  public CppModel setCompileHeaderModules(boolean compileHeaderModules) {
    this.compileHeaderModules = compileHeaderModules;
    return this;
  }
  
  /**
   * Whether to create actions for temps. This defaults to false.
   */
  public CppModel setSaveTemps(boolean maySaveTemps) {
    this.maySaveTemps = maySaveTemps;
    return this;
  }

  /**
   * Sets the compilation context, i.e. include directories and allowed header files inclusions.
   */
  public CppModel setContext(CppCompilationContext context) {
    this.context = context;
    return this;
  }

  /**
   * Adds a single source file to be compiled. Note that this should only be called for primary
   * compilation units, not for header files or files that are otherwise included.
   */
  public CppModel addSources(Iterable<Artifact> sourceFiles, Label sourceLabel) {
    for (Artifact sourceFile : sourceFiles) {
      this.sourceFiles.add(Pair.of(sourceFile, sourceLabel));
    }
    return this;
  }

  /**
   * Adds all the source files. Note that this should only be called for primary compilation units,
   * not for header files or files that are otherwise included.
   */
  public CppModel addSources(Iterable<Pair<Artifact, Label>> sources) {
    Iterables.addAll(this.sourceFiles, sources);
    return this;
  }

  /**
   * Adds the given copts.
   */
  public CppModel addCopts(Collection<String> copts) {
    this.copts.addAll(copts);
    return this;
  }

  /**
   * Sets the nocopts pattern. This is used to filter out flags from the system defined set of
   * flags. By default no filter is applied.
   */
  public CppModel setNoCopts(@Nullable Pattern nocopts) {
    this.nocopts = nocopts;
    return this;
  }

  /**
   * This can be used to specify additional include directories, without modifying the compilation
   * context.
   */
  public CppModel addAdditionalIncludes(Collection<PathFragment> additionalIncludes) {
    // TODO(bazel-team): Maybe this could be handled by the compilation context instead?
    this.additionalIncludes.addAll(additionalIncludes);
    return this;
  }

  /**
   * Adds the given linkopts to the optional dynamic library link command.
   */
  public CppModel addLinkopts(Collection<String> linkopts) {
    this.linkopts.addAll(linkopts);
    return this;
  }

  /**
   * Sets the link type used for the link actions. Note that only static links are supported at this
   * time.
   */
  public CppModel setLinkTargetType(LinkTargetType linkType) {
    this.linkType = linkType;
    return this;
  }

  public CppModel setNeverLink(boolean neverLink) {
    this.neverLink = neverLink;
    return this;
  }

  /**
   * Whether to allow interface dynamic libraries. Note that setting this to true only has an effect
   * if the configuration allows it. Defaults to false.
   */
  public CppModel setAllowInterfaceSharedObjects(boolean allowInterfaceSharedObjects) {
    // TODO(bazel-team): Set the default to true, and require explicit action to disable it.
    this.allowInterfaceSharedObjects = allowInterfaceSharedObjects;
    return this;
  }

  public CppModel setCreateDynamicLibrary(boolean createDynamicLibrary) {
    this.createDynamicLibrary = createDynamicLibrary;
    return this;
  }

  public CppModel setDynamicLibraryPath(PathFragment soImplFilename) {
    this.soImplFilename = soImplFilename;
    return this;
  }
  
  /**
   * Sets the feature configuration to be used for C/C++ actions. 
   */
  public CppModel setFeatureConfiguration(FeatureConfiguration featureConfiguration) {
    this.featureConfiguration = featureConfiguration;
    return this;
  }

  /**
   * @return the non-pic header module artifact for the current target.
   */
  public Artifact getHeaderModule(Artifact moduleMapArtifact) {
    PathFragment objectDir = CppHelper.getObjDirectory(ruleContext.getLabel());
    PathFragment outputName = objectDir.getRelative(
        semantics.getEffectiveSourcePath(moduleMapArtifact));
    return ruleContext.getRelatedArtifact(outputName, ".pcm");
  }

  /**
   * @return the pic header module artifact for the current target.
   */
  public Artifact getPicHeaderModule(Artifact moduleMapArtifact) {
    PathFragment objectDir = CppHelper.getObjDirectory(ruleContext.getLabel());
    PathFragment outputName = objectDir.getRelative(
        semantics.getEffectiveSourcePath(moduleMapArtifact));
    return ruleContext.getRelatedArtifact(outputName, ".pic.pcm");
  }

  /**
   * @return whether this target needs to generate pic actions.
   */
  public boolean getGeneratePicActions() {
    return CppHelper.usePic(ruleContext, false);
  }

  /**
   * @return whether this target needs to generate non-pic actions.
   */
  public boolean getGenerateNoPicActions() {
    return
        // If we always need pic for everything, then don't bother to create a no-pic action.
        (!CppHelper.usePic(ruleContext, true) || !CppHelper.usePic(ruleContext, false))
        // onlySingleOutput guarantees that the code is only ever linked into a dynamic library - so
        // we don't need a no-pic action even if linking into a binary would require it.
        && !((onlySingleOutput && getGeneratePicActions()));
  }

  /**
   * @return whether this target needs to generate a pic header module.
   */
  public boolean getGeneratesPicHeaderModule() {
    // TODO(bazel-team): Make sure cc_fake_binary works with header module support. 
    return compileHeaderModules && !fake && getGeneratePicActions();
  }

  /**
   * @return whether this target needs to generate a non-pic header module.
   */
  public boolean getGeratesNoPicHeaderModule() {
    return compileHeaderModules && !fake && getGenerateNoPicActions();
  }

  /**
   * Returns a {@code CppCompileActionBuilder} with the common fields for a C++ compile action
   * being initialized.
   */
  private CppCompileActionBuilder initializeCompileAction(Artifact sourceArtifact,
      Label sourceLabel) {
    CppCompileActionBuilder builder = createCompileActionBuilder(sourceArtifact, sourceLabel);
    if (nocopts != null) {
      builder.addNocopts(nocopts);
    }

    builder.setEnableLayeringCheck(enableLayeringCheck);
    builder.setCompileHeaderModules(compileHeaderModules);
    builder.setExtraSystemIncludePrefixes(additionalIncludes);
    builder.setFdoBuildStamp(CppHelper.getFdoBuildStamp(cppConfiguration));
    builder.setFeatureConfiguration(featureConfiguration);
    return builder;
  }

  /**
   * Constructs the C++ compiler actions. It generally creates one action for every specified source
   * file. It takes into account LIPO, fake-ness, coverage, and PIC, in addition to using the
   * settings specified on the current object. This method should only be called once.
   */
  public CcCompilationOutputs createCcCompileActions() {
    CcCompilationOutputs.Builder result = new CcCompilationOutputs.Builder();
    Preconditions.checkNotNull(context);
    AnalysisEnvironment env = ruleContext.getAnalysisEnvironment();
    PathFragment objectDir = CppHelper.getObjDirectory(ruleContext.getLabel());
    
    if (compileHeaderModules) {
      Artifact moduleMapArtifact = context.getCppModuleMap().getArtifact();
      Label moduleMapLabel = Label.parseAbsoluteUnchecked(context.getCppModuleMap().getName());
      PathFragment outputName = getObjectOutputPath(moduleMapArtifact, objectDir);
      CppCompileActionBuilder builder = initializeCompileAction(moduleMapArtifact, moduleMapLabel);

      // A header module compile action is just like a normal compile action, but:
      // - the compiled source file is the module map
      // - it creates a header module (.pcm file).
      createSourceAction(outputName, result, env, moduleMapArtifact, builder, ".pcm", ".pcm.d");
    }

    for (Pair<Artifact, Label> source : sourceFiles) {
      Artifact sourceArtifact = source.getFirst();
      Label sourceLabel = source.getSecond();
      PathFragment outputName = getObjectOutputPath(sourceArtifact, objectDir);
      CppCompileActionBuilder builder = initializeCompileAction(sourceArtifact, sourceLabel);
      
      if (CppFileTypes.CPP_HEADER.matches(source.first.getExecPath())) {
        createHeaderAction(outputName, result, env, builder);
      } else {
        createSourceAction(outputName, result, env, sourceArtifact, builder, ".o", ".d");
      }
    }

    compilationOutputs = result.build();
    return compilationOutputs;
  }

  private void createHeaderAction(PathFragment outputName, Builder result, AnalysisEnvironment env,
      CppCompileActionBuilder builder) {
    builder
        .setOutputFile(ruleContext.getRelatedArtifact(outputName, ".h.processed"))
        .setDotdFile(outputName, ".h.d", ruleContext)
        // If we generate pic actions, we prefer the header actions to use the pic artifacts.
        .setPicMode(this.getGeneratePicActions());
    semantics.finalizeCompileActionBuilder(ruleContext, builder);
    CppCompileAction compileAction = builder.build();
    env.registerAction(compileAction);
    Artifact tokenFile = compileAction.getOutputFile();
    result.addHeaderTokenFile(tokenFile);
  }

  private void createSourceAction(PathFragment outputName,
      CcCompilationOutputs.Builder result,
      AnalysisEnvironment env,
      Artifact sourceArtifact,
      CppCompileActionBuilder builder,
      String outputExtension,
      String dependencyFileExtension) {
    PathFragment ccRelativeName = semantics.getEffectiveSourcePath(sourceArtifact);
    LipoContextProvider lipoProvider = null;
    if (cppConfiguration.isLipoOptimization()) {
      // TODO(bazel-team): we shouldn't be needing this, merging context with the binary
      // is a superset of necessary information.
      lipoProvider = Preconditions.checkNotNull(CppHelper.getLipoContextProvider(ruleContext),
          outputName);
      builder.setContext(CppCompilationContext.mergeForLipo(lipoProvider.getLipoContext(),
          context));
    }
    if (fake) {
      // For cc_fake_binary, we only create a single fake compile action. It's
      // not necessary to use -fPIC for negative compilation tests, and using
      // .pic.o files in cc_fake_binary would break existing uses of
      // cc_fake_binary.
      Artifact outputFile = ruleContext.getRelatedArtifact(outputName, outputExtension);
      PathFragment tempOutputName =
          FileSystemUtils.replaceExtension(outputFile.getExecPath(), ".temp" + outputExtension);
      builder
          .setOutputFile(outputFile)
          .setDotdFile(outputName, dependencyFileExtension, ruleContext)
          .setTempOutputFile(tempOutputName);
      semantics.finalizeCompileActionBuilder(ruleContext, builder);
      CppCompileAction action = builder.build();
      env.registerAction(action);
      result.addObjectFile(action.getOutputFile());
    } else {
      boolean generatePicAction = getGeneratePicActions();
      // If we always need pic for everything, then don't bother to create a no-pic action.
      boolean generateNoPicAction = getGenerateNoPicActions();
      Preconditions.checkState(generatePicAction || generateNoPicAction);

      // Create PIC compile actions (same as non-PIC, but use -fPIC and
      // generate .pic.o, .pic.d, .pic.gcno instead of .o, .d, .gcno.)
      if (generatePicAction) {
        CppCompileActionBuilder picBuilder =
            copyAsPicBuilder(builder, outputName, outputExtension, dependencyFileExtension);
        cppConfiguration.getFdoSupport().configureCompilation(picBuilder, ruleContext, env,
            ruleContext.getLabel(), ccRelativeName, nocopts, /*usePic=*/true,
            lipoProvider);

        if (maySaveTemps) {
          result.addTemps(
              createTempsActions(sourceArtifact, outputName, picBuilder, /*usePic=*/true));
        }

        if (isCodeCoverageEnabled()) {
          picBuilder.setGcnoFile(ruleContext.getRelatedArtifact(outputName, ".pic.gcno"));
        }

        semantics.finalizeCompileActionBuilder(ruleContext, picBuilder);
        CppCompileAction picAction = picBuilder.build();
        env.registerAction(picAction);
        result.addPicObjectFile(picAction.getOutputFile());
        if (picAction.getDwoFile() != null) {
          // Host targets don't produce .dwo files.
          result.addPicDwoFile(picAction.getDwoFile());
        }
        if (cppConfiguration.isLipoContextCollector() && !generateNoPicAction) {
          result.addLipoScannable(picAction);
        }
      }

      if (generateNoPicAction) {
        builder
            .setOutputFile(ruleContext.getRelatedArtifact(outputName, outputExtension))
            .setDotdFile(outputName, dependencyFileExtension, ruleContext);
        // Create non-PIC compile actions
        cppConfiguration.getFdoSupport().configureCompilation(builder, ruleContext, env,
            ruleContext.getLabel(), ccRelativeName, nocopts, /*usePic=*/false,
            lipoProvider);

        if (maySaveTemps) {
          result.addTemps(
              createTempsActions(sourceArtifact, outputName, builder, /*usePic=*/false));
        }

        if (!cppConfiguration.isLipoOptimization() && isCodeCoverageEnabled()) {
          builder.setGcnoFile(ruleContext.getRelatedArtifact(outputName, ".gcno"));
        }

        semantics.finalizeCompileActionBuilder(ruleContext, builder);
        CppCompileAction compileAction = builder.build();
        env.registerAction(compileAction);
        Artifact objectFile = compileAction.getOutputFile();
        result.addObjectFile(objectFile);
        if (compileAction.getDwoFile() != null) {
          // Host targets don't produce .dwo files.
          result.addDwoFile(compileAction.getDwoFile());
        }
        if (cppConfiguration.isLipoContextCollector()) {
          result.addLipoScannable(compileAction);
        }
      }
    }
  }

  /**
   * Constructs the C++ linker actions. It generally generates two actions, one for a static library
   * and one for a dynamic library. If PIC is required for shared libraries, but not for binaries,
   * it additionally creates a third action to generate a PIC static library.
   *
   * <p>For dynamic libraries, this method can additionally create an interface shared library that
   * can be used for linking, but doesn't contain any executable code. This increases the number of
   * cache hits for link actions. Call {@link #setAllowInterfaceSharedObjects(boolean)} to enable
   * this behavior.
   */
  public CcLinkingOutputs createCcLinkActions(CcCompilationOutputs ccOutputs) {
    // For now only handle static links. Note that the dynamic library link below ignores linkType.
    // TODO(bazel-team): Either support non-static links or move this check to setLinkType().
    Preconditions.checkState(linkType.isStaticLibraryLink(), "can only handle static links");

    CcLinkingOutputs.Builder result = new CcLinkingOutputs.Builder();
    if (cppConfiguration.isLipoContextCollector()) {
      // Don't try to create LIPO link actions in collector mode,
      // because it needs some data that's not available at this point.
      return result.build();
    }

    AnalysisEnvironment env = ruleContext.getAnalysisEnvironment();
    boolean usePicForBinaries = CppHelper.usePic(ruleContext, true);
    boolean usePicForSharedLibs = CppHelper.usePic(ruleContext, false);

    // Create static library (.a). The linkType only reflects whether the library is alwayslink or
    // not. The PIC-ness is determined by whether we need to use PIC or not. There are three cases
    // for (usePicForSharedLibs usePicForBinaries):
    //
    // (1) (false false) -> no pic code
    // (2) (true false)  -> shared libraries as pic, but not binaries
    // (3) (true true)   -> both shared libraries and binaries as pic
    //
    // In case (3), we always need PIC, so only create one static library containing the PIC object
    // files. The name therefore does not match the content.
    //
    // Presumably, it is done this way because the .a file is an implicit output of every cc_library
    // rule, so we can't use ".pic.a" that in the always-PIC case.
    PathFragment linkedFileName = CppHelper.getLinkedFilename(ruleContext, linkType);
    CppLinkAction maybePicAction = newLinkActionBuilder(linkedFileName)
        .addNonLibraryInputs(ccOutputs.getObjectFiles(usePicForBinaries))
        .addNonLibraryInputs(ccOutputs.getHeaderTokenFiles())
        .setLinkType(linkType)
        .setLinkStaticness(LinkStaticness.FULLY_STATIC)
        .build();
    env.registerAction(maybePicAction);
    result.addStaticLibrary(maybePicAction.getOutputLibrary());

    // Create a second static library (.pic.a). Only in case (2) do we need both PIC and non-PIC
    // static libraries. In that case, the first static library contains the non-PIC code, and this
    // one contains the PIC code, so the names match the content.
    if (!usePicForBinaries && usePicForSharedLibs) {
      LinkTargetType picLinkType = (linkType == LinkTargetType.ALWAYS_LINK_STATIC_LIBRARY)
          ? LinkTargetType.ALWAYS_LINK_PIC_STATIC_LIBRARY
          : LinkTargetType.PIC_STATIC_LIBRARY;

      PathFragment picFileName = CppHelper.getLinkedFilename(ruleContext, picLinkType);
      CppLinkAction picAction = newLinkActionBuilder(picFileName)
          .addNonLibraryInputs(ccOutputs.getObjectFiles(true))
          .addNonLibraryInputs(ccOutputs.getHeaderTokenFiles())
          .setLinkType(picLinkType)
          .setLinkStaticness(LinkStaticness.FULLY_STATIC)
          .build();
      env.registerAction(picAction);
      result.addPicStaticLibrary(picAction.getOutputLibrary());
    }

    if (!createDynamicLibrary) {
      return result.build();
    }

    // Create dynamic library.
    if (soImplFilename == null) {
      soImplFilename = CppHelper.getLinkedFilename(ruleContext, LinkTargetType.DYNAMIC_LIBRARY);
    }
    List<String> sonameLinkopts = ImmutableList.of();
    PathFragment soInterfaceFilename = null;
    if (cppConfiguration.useInterfaceSharedObjects() && allowInterfaceSharedObjects) {
      soInterfaceFilename =
          CppHelper.getLinkedFilename(ruleContext, LinkTargetType.INTERFACE_DYNAMIC_LIBRARY);
      Artifact dynamicLibrary = env.getDerivedArtifact(
          soImplFilename, configuration.getBinDirectory());
      sonameLinkopts = ImmutableList.of("-Wl,-soname=" +
          SolibSymlinkAction.getDynamicLibrarySoname(dynamicLibrary.getRootRelativePath(), false));
    }

    // Should we also link in any libraries that this library depends on?
    // That is required on some systems...
    CppLinkAction action = newLinkActionBuilder(soImplFilename)
        .setInterfaceOutputPath(soInterfaceFilename)
        .addNonLibraryInputs(ccOutputs.getObjectFiles(usePicForSharedLibs))
        .addNonLibraryInputs(ccOutputs.getHeaderTokenFiles())
        .setLinkType(LinkTargetType.DYNAMIC_LIBRARY)
        .setLinkStaticness(LinkStaticness.DYNAMIC)
        .addLinkopts(linkopts)
        .addLinkopts(sonameLinkopts)
        .setRuntimeInputs(
            CppHelper.getToolchain(ruleContext).getDynamicRuntimeLinkMiddleman(),
            CppHelper.getToolchain(ruleContext).getDynamicRuntimeLinkInputs())
        .build();
    env.registerAction(action);

    LibraryToLink dynamicLibrary = action.getOutputLibrary();
    LibraryToLink interfaceLibrary = action.getInterfaceOutputLibrary();
    if (interfaceLibrary == null) {
      interfaceLibrary = dynamicLibrary;
    }

    // If shared library has neverlink=1, then leave it untouched. Otherwise,
    // create a mangled symlink for it and from now on reference it through
    // mangled name only.
    if (neverLink) {
      result.addDynamicLibrary(interfaceLibrary);
      result.addExecutionDynamicLibrary(dynamicLibrary);
    } else {
      LibraryToLink libraryLink = SolibSymlinkAction.getDynamicLibrarySymlink(
          ruleContext, interfaceLibrary.getArtifact(), false, false,
          ruleContext.getConfiguration());
      result.addDynamicLibrary(libraryLink);
      LibraryToLink implLibraryLink = SolibSymlinkAction.getDynamicLibrarySymlink(
          ruleContext, dynamicLibrary.getArtifact(), false, false,
          ruleContext.getConfiguration());
      result.addExecutionDynamicLibrary(implLibraryLink);
    }
    return result.build();
  }

  private CppLinkAction.Builder newLinkActionBuilder(PathFragment outputPath) {
    return new CppLinkAction.Builder(ruleContext, outputPath)
        .setCrosstoolInputs(CppHelper.getToolchain(ruleContext).getLink())
        .addNonLibraryInputs(context.getCompilationPrerequisites());
  }

  /**
   * Returns the output artifact path relative to the object directory.
   */
  private PathFragment getObjectOutputPath(Artifact source, PathFragment objectDirectory) {
    return objectDirectory.getRelative(semantics.getEffectiveSourcePath(source));
  }

  /**
   * Creates a basic cpp compile action builder for source file. Configures options,
   * crosstool inputs, output and dotd file names, compilation context and copts.
   */
  private CppCompileActionBuilder createCompileActionBuilder(
      Artifact source, Label label) {
    CppCompileActionBuilder builder = new CppCompileActionBuilder(
        ruleContext, source, label);

    builder
        .setContext(context)
        .addCopts(copts);
    return builder;
  }

  /**
   * Creates cpp PIC compile action builder from the given builder by adding necessary copt and
   * changing output and dotd file names.
   */
  private CppCompileActionBuilder copyAsPicBuilder(CppCompileActionBuilder builder,
      PathFragment outputName, String outputExtension, String dependencyFileExtension) {
    CppCompileActionBuilder picBuilder = new CppCompileActionBuilder(builder);
    picBuilder
        .setPicMode(true)
        .setOutputFile(ruleContext.getRelatedArtifact(outputName, ".pic" + outputExtension))
        .setDotdFile(outputName, ".pic" + dependencyFileExtension, ruleContext);
    return picBuilder;
  }

  /**
   * Create the actions for "--save_temps".
   */
  private ImmutableList<Artifact> createTempsActions(Artifact source, PathFragment outputName,
      CppCompileActionBuilder builder, boolean usePic) {
    if (!cppConfiguration.getSaveTemps()) {
      return ImmutableList.of();
    }

    String path = source.getFilename();
    boolean isCFile = CppFileTypes.C_SOURCE.matches(path);
    boolean isCppFile = CppFileTypes.CPP_SOURCE.matches(path);

    if (!isCFile && !isCppFile) {
      return ImmutableList.of();
    }

    String iExt = isCFile ? ".i" : ".ii";
    String picExt = usePic ? ".pic" : "";
    CppCompileActionBuilder dBuilder = new CppCompileActionBuilder(builder);
    CppCompileActionBuilder sdBuilder = new CppCompileActionBuilder(builder);

    dBuilder
        .setOutputFile(ruleContext.getRelatedArtifact(outputName, picExt + iExt))
        .setDotdFile(outputName, picExt + iExt + ".d", ruleContext);
    semantics.finalizeCompileActionBuilder(ruleContext, dBuilder);
    CppCompileAction dAction = dBuilder.build();
    ruleContext.registerAction(dAction);

    sdBuilder
        .setOutputFile(ruleContext.getRelatedArtifact(outputName, picExt + ".s"))
        .setDotdFile(outputName, picExt + ".s.d", ruleContext);
    semantics.finalizeCompileActionBuilder(ruleContext, sdBuilder);
    CppCompileAction sdAction = sdBuilder.build();
    ruleContext.registerAction(sdAction);
    return ImmutableList.of(
        dAction.getOutputFile(),
        sdAction.getOutputFile());
  }

  /**
   * Returns true iff code coverage is enabled for the given target.
   */
  private boolean isCodeCoverageEnabled() {
    if (configuration.isCodeCoverageEnabled()) {
      final RegexFilter filter = configuration.getInstrumentationFilter();
      // If rule is matched by the instrumentation filter, enable instrumentation
      if (filter.isIncluded(ruleContext.getLabel().toString())) {
        return true;
      }
      // At this point the rule itself is not matched by the instrumentation filter. However, we
      // might still want to instrument C++ rules if one of the targets listed in "deps" is
      // instrumented and, therefore, can supply header files that we would want to collect code
      // coverage for. For example, think about cc_test rule that tests functionality defined in a
      // header file that is supplied by the cc_library.
      //
      // Note that we only check direct prerequisites and not the transitive closure. This is done
      // for two reasons:
      // a) It is a good practice to declare libraries which you directly rely on. Including headers
      //    from a library hidden deep inside the transitive closure makes build dependencies less
      //    readable and can lead to unexpected breakage.
      // b) Traversing the transitive closure for each C++ compile action would require more complex
      //    implementation (with caching results of this method) to avoid O(N^2) slowdown.
      if (ruleContext.getRule().isAttrDefined("deps", Type.LABEL_LIST)) {
        for (TransitiveInfoCollection dep : ruleContext.getPrerequisites("deps", Mode.TARGET)) {
          if (dep.getProvider(CppCompilationContext.class) != null
              && filter.isIncluded(dep.getLabel().toString())) {
            return true;
          }
        }
      }
    }
    return false;
  }
}
