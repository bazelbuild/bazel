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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.rules.cpp.CcCompilationOutputs.Builder;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.Link.LinkStaticness;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
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

  // link model
  private final List<String> linkopts = new ArrayList<>();
  private LinkTargetType linkType = LinkTargetType.STATIC_LIBRARY;
  private boolean neverLink;
  private boolean allowInterfaceSharedObjects;
  private boolean createDynamicLibrary = true;
  private Artifact soImplArtifact;
  private FeatureConfiguration featureConfiguration;

  public CppModel(RuleContext ruleContext, CppSemantics semantics) {
    this.ruleContext = Preconditions.checkNotNull(ruleContext);
    this.semantics = semantics;
    configuration = ruleContext.getConfiguration();
    cppConfiguration = ruleContext.getFragment(CppConfiguration.class);
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

  public CppModel setDynamicLibrary(Artifact soImplFilename) {
    this.soImplArtifact = soImplFilename;
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
  private boolean getGeneratePicActions() {
    return CppHelper.usePic(ruleContext, false);
  }

  /**
   * @return whether this target needs to generate non-pic actions.
   */
  private boolean getGenerateNoPicActions() {
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
    return featureConfiguration.isEnabled(CppRuleClasses.HEADER_MODULES) && !fake
        && getGeneratePicActions();
  }

  /**
   * @return whether this target needs to generate a non-pic header module.
   */
  public boolean getGeneratesNoPicHeaderModule() {
    return featureConfiguration.isEnabled(CppRuleClasses.HEADER_MODULES) && !fake
        && getGenerateNoPicActions();
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

    builder.setExtraSystemIncludePrefixes(additionalIncludes);
    builder.setFdoBuildStamp(CppHelper.getFdoBuildStamp(cppConfiguration));
    builder.setFeatureConfiguration(featureConfiguration);
    return builder;
  }

  /**
   * Get the safe path strings for a list of paths to use in the build variables.
   */
  private Collection<String> getSafePathStrings(Collection<PathFragment> paths) {
    ImmutableSet.Builder<String> result = ImmutableSet.builder();
    for (PathFragment path : paths) {
      result.add(path.getSafePathString());
    }
    return result.build();
  }

  /**
   * Select .pcm inputs to pass on the command line depending on whether we are in pic or non-pic
   * mode.
   */
  private Collection<String> getHeaderModulePaths(CppCompileActionBuilder builder,
      boolean usePic) {
    Collection<String> result = new LinkedHashSet<>();
    NestedSet<Artifact> artifacts = featureConfiguration.isEnabled(
        CppRuleClasses.HEADER_MODULE_INCLUDES_DEPENDENCIES)
        ? builder.getContext().getTopLevelHeaderModules()
        : builder.getContext().getAdditionalInputs();
    for (Artifact artifact : artifacts) {
      String filename = artifact.getFilename();
      if (!filename.endsWith(".pcm")) {
        continue;
      }
      // Depending on whether this specific compile action is pic or non-pic, select the
      // corresponding header modules. Note that the compilation context might give us both
      // from targets that are built in both modes.
      if (usePic == filename.endsWith(".pic.pcm")) {
        result.add(artifact.getExecPathString());
      }
    }
    return result;
  }

  private void setupBuildVariables(
      CppCompileActionBuilder builder,
      boolean usePic,
      PathFragment ccRelativeName,
      Artifact gcnoFile) {
    CcToolchainFeatures.Variables.Builder buildVariables =
        new CcToolchainFeatures.Variables.Builder();
    
    // TODO(bazel-team): Pull out string constants for all build variables.

    CppModuleMap cppModuleMap = context.getCppModuleMap();
    if (featureConfiguration.isEnabled(CppRuleClasses.MODULE_MAPS) && cppModuleMap != null) {
      // If the feature is enabled and cppModuleMap is null, we are about to fail during analysis
      // in any case, but don't crash.
      buildVariables.addVariable("module_name", cppModuleMap.getName());
      buildVariables.addVariable("module_map_file",
          cppModuleMap.getArtifact().getExecPathString());
      CcToolchainFeatures.Variables.NestedSequence.Builder sequence =
          new CcToolchainFeatures.Variables.NestedSequence.Builder();
      for (Artifact artifact : context.getDirectModuleMaps()) {
        sequence.addValue(artifact.getExecPathString());
      }
      buildVariables.addSequence("dependent_module_map_files", sequence.build());
    }
    if (featureConfiguration.isEnabled(CppRuleClasses.USE_HEADER_MODULES)) {
      buildVariables.addSequenceVariable("module_files", getHeaderModulePaths(builder, usePic));
    }
    if (featureConfiguration.isEnabled(CppRuleClasses.INCLUDE_PATHS)) {
      buildVariables.addSequenceVariable("include_paths",
          getSafePathStrings(context.getIncludeDirs()));
      buildVariables.addSequenceVariable("quote_include_paths",
          getSafePathStrings(context.getQuoteIncludeDirs()));
      buildVariables.addSequenceVariable("system_include_paths",
          getSafePathStrings(context.getSystemIncludeDirs()));
    }

    if (ccRelativeName != null) {
      cppConfiguration.getFdoSupport().configureCompilation(builder, buildVariables, ruleContext,
          ccRelativeName, usePic, featureConfiguration);
    }
    if (gcnoFile != null) {
      buildVariables.addVariable("gcov_gcno_file", gcnoFile.getExecPathString());
    }

    CcToolchainFeatures.Variables variables = buildVariables.build();
    builder.setVariables(variables);
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
    
    if (featureConfiguration.isEnabled(CppRuleClasses.HEADER_MODULES)) {
      Artifact moduleMapArtifact = context.getCppModuleMap().getArtifact();
      Label moduleMapLabel = Label.parseAbsoluteUnchecked(context.getCppModuleMap().getName());
      PathFragment outputName = getObjectOutputPath(moduleMapArtifact, objectDir);
      CppCompileActionBuilder builder = initializeCompileAction(moduleMapArtifact, moduleMapLabel);

      // A header module compile action is just like a normal compile action, but:
      // - the compiled source file is the module map
      // - it creates a header module (.pcm file).
      createSourceAction(outputName, result, env, moduleMapArtifact, builder, ".pcm", ".pcm.d",
          /*addObject=*/false);
    }

    for (Pair<Artifact, Label> source : sourceFiles) {
      Artifact sourceArtifact = source.getFirst();
      Label sourceLabel = source.getSecond();
      PathFragment outputName = getObjectOutputPath(sourceArtifact, objectDir);
      CppCompileActionBuilder builder = initializeCompileAction(sourceArtifact, sourceLabel);
      
      if (CppFileTypes.CPP_HEADER.matches(source.first.getExecPath())) {
        createHeaderAction(outputName, result, env, builder);
      } else {
        createSourceAction(outputName, result, env, sourceArtifact, builder, ".o", ".d",
            /*addObject=*/true);
      }
    }

    compilationOutputs = result.build();
    return compilationOutputs;
  }

  private void createHeaderAction(PathFragment outputName, Builder result, AnalysisEnvironment env,
      CppCompileActionBuilder builder) {
    builder
        .setOutputFile(ruleContext.getRelatedArtifact(outputName, ".h.processed"))
        .setDotdFile(outputName, ".h.d")
        // If we generate pic actions, we prefer the header actions to use the pic artifacts.
        .setPicMode(this.getGeneratePicActions());
    setupBuildVariables(builder, this.getGeneratePicActions(), null, null);
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
      String dependencyFileExtension,
      boolean addObject) {
    PathFragment ccRelativeName = semantics.getEffectiveSourcePath(sourceArtifact);
    if (cppConfiguration.isLipoOptimization()) {
      // TODO(bazel-team): we shouldn't be needing this, merging context with the binary
      // is a superset of necessary information.
      LipoContextProvider lipoProvider =
          Preconditions.checkNotNull(CppHelper.getLipoContextProvider(ruleContext), outputName);
      builder.setContext(CppCompilationContext.mergeForLipo(lipoProvider.getLipoContext(),
          context));
    }
    boolean generatePicAction = getGeneratePicActions();
    // If we always need pic for everything, then don't bother to create a no-pic action.
    boolean generateNoPicAction = getGenerateNoPicActions();
    Preconditions.checkState(generatePicAction || generateNoPicAction);
    if (fake) {
      boolean usePic = !generateNoPicAction;
      createFakeSourceAction(outputName, result, env, builder, outputExtension,
          dependencyFileExtension, addObject, ccRelativeName, usePic);
    } else {
      // Create PIC compile actions (same as non-PIC, but use -fPIC and
      // generate .pic.o, .pic.d, .pic.gcno instead of .o, .d, .gcno.)
      if (generatePicAction) {
        CppCompileActionBuilder picBuilder =
            copyAsPicBuilder(builder, outputName, outputExtension, dependencyFileExtension);
        Artifact gcnoFile =
            isCodeCoverageEnabled()
                ? ruleContext.getRelatedArtifact(outputName, ".pic.gcno")
                : null;
        if (gcnoFile != null) {
          picBuilder.setGcnoFile(gcnoFile);
        }
        setupBuildVariables(picBuilder, /*usePic=*/ true, ccRelativeName, gcnoFile);

        if (maySaveTemps) {
          result.addTemps(
              createTempsActions(sourceArtifact, outputName, picBuilder, /*usePic=*/true,
                ccRelativeName));
        }

        semantics.finalizeCompileActionBuilder(ruleContext, picBuilder);
        CppCompileAction picAction = picBuilder.build();
        env.registerAction(picAction);
        if (addObject) {
          result.addPicObjectFile(picAction.getOutputFile());

          if (featureConfiguration.isEnabled(CppRuleClasses.THIN_LTO)
              && CppFileTypes.LTO_SOURCE.matches(sourceArtifact.getFilename())) {
            result.addLTOBitcodeFile(picAction.getOutputFile());
          }
        }
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
            .setDotdFile(outputName, dependencyFileExtension);
        // Create non-PIC compile actions
        Artifact gcnoFile =
            !cppConfiguration.isLipoOptimization() && isCodeCoverageEnabled()
                ? ruleContext.getRelatedArtifact(outputName, ".gcno")
                : null;
        if (gcnoFile != null) {
          builder.setGcnoFile(gcnoFile);
        }
        setupBuildVariables(builder, /*usePic=*/ false, ccRelativeName, gcnoFile);

        if (maySaveTemps) {
          result.addTemps(
              createTempsActions(
                  sourceArtifact,
                  outputName,
                  builder,
                  /*usePic=*/ false,
                  ccRelativeName));
        }

        semantics.finalizeCompileActionBuilder(ruleContext, builder);
        CppCompileAction compileAction = builder.build();
        env.registerAction(compileAction);
        Artifact objectFile = compileAction.getOutputFile();
        if (addObject) {
          result.addObjectFile(objectFile);
          if (featureConfiguration.isEnabled(CppRuleClasses.THIN_LTO)
              && CppFileTypes.LTO_SOURCE.matches(sourceArtifact.getFilename())) {
            result.addLTOBitcodeFile(objectFile);
          }
        }
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

  private void createFakeSourceAction(PathFragment outputName, CcCompilationOutputs.Builder result,
      AnalysisEnvironment env, CppCompileActionBuilder builder, String outputExtension,
      String dependencyFileExtension, boolean addObject, PathFragment ccRelativeName,
      boolean usePic) {
    if (usePic) {
      outputExtension = ".pic" + outputExtension;
      dependencyFileExtension = ".pic" + dependencyFileExtension;
    }
    Artifact outputFile = ruleContext.getRelatedArtifact(outputName, outputExtension);
    PathFragment tempOutputName =
        FileSystemUtils.replaceExtension(
            outputFile.getExecPath(), ".temp" + outputExtension, outputExtension);
    builder
        .setPicMode(usePic)
        .setOutputFile(outputFile)
        .setDotdFile(outputName, dependencyFileExtension)
        .setTempOutputFile(tempOutputName);
    setupBuildVariables(builder, getGeneratePicActions(), ccRelativeName, null);
    semantics.finalizeCompileActionBuilder(ruleContext, builder);
    CppCompileAction action = builder.build();
    env.registerAction(action);
    if (addObject) {
      if (usePic) {
        result.addPicObjectFile(action.getOutputFile());
      } else {
        result.addObjectFile(action.getOutputFile());
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
    Artifact linkedArtifact = CppHelper.getLinkedArtifact(ruleContext, linkType);
    CppLinkAction maybePicAction =
        newLinkActionBuilder(linkedArtifact)
            .addNonLibraryInputs(ccOutputs.getObjectFiles(usePicForBinaries))
            .addNonLibraryInputs(ccOutputs.getHeaderTokenFiles())
            .addLTOBitcodeFiles(ccOutputs.getLtoBitcodeFiles())
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

      Artifact picArtifact = CppHelper.getLinkedArtifact(ruleContext, picLinkType);
      CppLinkAction picAction =
          newLinkActionBuilder(picArtifact)
              .addNonLibraryInputs(ccOutputs.getObjectFiles(true))
              .addNonLibraryInputs(ccOutputs.getHeaderTokenFiles())
              .addLTOBitcodeFiles(ccOutputs.getLtoBitcodeFiles())
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
    Artifact soImpl;
    if (soImplArtifact == null) {
      soImpl = CppHelper.getLinkedArtifact(ruleContext, LinkTargetType.DYNAMIC_LIBRARY);
    } else {
      soImpl = soImplArtifact;
    }

    List<String> sonameLinkopts = ImmutableList.of();
    Artifact soInterface = null;
    if (cppConfiguration.useInterfaceSharedObjects() && allowInterfaceSharedObjects) {
      soInterface =
          CppHelper.getLinkedArtifact(ruleContext, LinkTargetType.INTERFACE_DYNAMIC_LIBRARY);
      sonameLinkopts = ImmutableList.of("-Wl,-soname=" +
          SolibSymlinkAction.getDynamicLibrarySoname(soImpl.getRootRelativePath(), false));
    }

    // Should we also link in any libraries that this library depends on?
    // That is required on some systems...
    CppLinkAction action = newLinkActionBuilder(soImpl)
        .setInterfaceOutput(soInterface)
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

  private CppLinkAction.Builder newLinkActionBuilder(Artifact outputArtifact) {
    return new CppLinkAction.Builder(ruleContext, outputArtifact)
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
        .setDotdFile(outputName, ".pic" + dependencyFileExtension);
    return picBuilder;
  }

  /**
   * Create the actions for "--save_temps".
   */
  private ImmutableList<Artifact> createTempsActions(Artifact source, PathFragment outputName,
      CppCompileActionBuilder builder, boolean usePic, PathFragment ccRelativeName) {
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
    setupBuildVariables(dBuilder, usePic, ccRelativeName, null);
    CppCompileActionBuilder sdBuilder = new CppCompileActionBuilder(builder);
    setupBuildVariables(sdBuilder, usePic, ccRelativeName, null);

    dBuilder
        .setOutputFile(ruleContext.getRelatedArtifact(outputName, picExt + iExt))
        .setDotdFile(outputName, picExt + iExt + ".d");
    semantics.finalizeCompileActionBuilder(ruleContext, dBuilder);
    CppCompileAction dAction = dBuilder.build();
    ruleContext.registerAction(dAction);

    sdBuilder
        .setOutputFile(ruleContext.getRelatedArtifact(outputName, picExt + ".s"))
        .setDotdFile(outputName, picExt + ".s.d");
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
      if (ruleContext.getRule().isAttrDefined("deps", BuildType.LABEL_LIST)) {
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
