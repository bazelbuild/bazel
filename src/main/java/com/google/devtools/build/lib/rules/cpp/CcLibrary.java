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

import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.List;

/**
 * A ConfiguredTarget for <code>cc_library</code> rules.
 */
public abstract class CcLibrary implements RuleConfiguredTargetFactory {

  private final CppSemantics semantics;

  protected CcLibrary(CppSemantics semantics) {
    this.semantics = semantics;
  }

  // These file extensions don't generate object files.
  private static final FileTypeSet NO_OBJECT_GENERATING_FILETYPES = FileTypeSet.of(
      CppFileTypes.CPP_HEADER, CppFileTypes.ARCHIVE, CppFileTypes.PIC_ARCHIVE,
      CppFileTypes.ALWAYS_LINK_LIBRARY, CppFileTypes.ALWAYS_LINK_PIC_LIBRARY,
      CppFileTypes.SHARED_LIBRARY);

  private static final Predicate<LibraryToLink> PIC_STATIC_FILTER = new Predicate<LibraryToLink>() {
    @Override
    public boolean apply(LibraryToLink input) {
      String name = input.getArtifact().getExecPath().getBaseName();
      return !name.endsWith(".nopic.a") && !name.endsWith(".nopic.lo");
    }
  };

  private static Runfiles collectRunfiles(RuleContext context,
      CcLinkingOutputs ccLinkingOutputs,
      boolean neverLink, boolean addDynamicRuntimeInputArtifactsToRunfiles,
      boolean linkingStatically) {
    Runfiles.Builder builder = new Runfiles.Builder(context.getWorkspaceName());

    // neverlink= true creates a library that will never be linked into any binary that depends on
    // it, but instead be loaded as an extension. So we need the dynamic library for this in the
    // runfiles.
    builder.addArtifacts(ccLinkingOutputs.getLibrariesForRunfiles(linkingStatically && !neverLink));
    builder.add(context, CppRunfilesProvider.runfilesFunction(linkingStatically));

    builder.addDataDeps(context);

    if (addDynamicRuntimeInputArtifactsToRunfiles) {
      builder.addTransitiveArtifacts(CppHelper.getToolchain(context).getDynamicRuntimeLinkInputs());
    }
    return builder.build();
  }

  @Override
  public ConfiguredTarget create(RuleContext context) {
    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(context);
    LinkTargetType linkType = getStaticLinkType(context);
    boolean linkStatic = context.attributes().get("linkstatic", Type.BOOLEAN);
    init(semantics, context, builder, linkType,
        /*neverLink =*/ false,
        linkStatic,
        /*collectLinkstamp =*/ true,
        /*addDynamicRuntimeInputArtifactsToRunfiles =*/ false);
    return builder.build();
  }

  public static void init(CppSemantics semantics, RuleContext ruleContext,
      RuleConfiguredTargetBuilder targetBuilder, LinkTargetType linkType,
      boolean neverLink,
      boolean linkStatic,
      boolean collectLinkstamp,
      boolean addDynamicRuntimeInputArtifactsToRunfiles) {
    FeatureConfiguration featureConfiguration = CcCommon.configureFeatures(ruleContext);
    final CcCommon common = new CcCommon(ruleContext);
    PrecompiledFiles precompiledFiles = new PrecompiledFiles(ruleContext);

    CcLibraryHelper helper =
        new CcLibraryHelper(ruleContext, semantics, featureConfiguration)
            .fromCommon(common)
            .addLinkopts(common.getLinkopts())
            .addSources(common.getSources())
            .addPublicHeaders(common.getHeaders())
            .enableCcNativeLibrariesProvider()
            .enableCompileProviders()
            .enableInterfaceSharedObjects()
            // Generate .a and .so outputs even without object files to fulfill the rule class contract
            // wrt. implicit output files, if the contract says so. Behavior here differs between Bazel
            // and Blaze.
            .setGenerateLinkActionsIfEmpty(
                ruleContext.getRule().getRuleClassObject().getImplicitOutputsFunction()
                    != ImplicitOutputsFunction.NONE)
            .setLinkType(linkType)
            .setNeverLink(neverLink)
            .addPrecompiledFiles(precompiledFiles);

    if (collectLinkstamp) {
      helper.addLinkstamps(ruleContext.getPrerequisites("linkstamp", Mode.TARGET));
    }

    Artifact soImplArtifact = null;
    boolean supportsDynamicLinker =
        ruleContext.getFragment(CppConfiguration.class).supportsDynamicLinker();
    boolean createDynamicLibrary =
        !linkStatic && appearsToHaveObjectFiles(ruleContext.attributes()) && supportsDynamicLinker;
    if (ruleContext.getRule().isAttrDefined("outs", Type.STRING_LIST)) {
      List<String> outs = ruleContext.attributes().get("outs", Type.STRING_LIST);
      if (outs.size() > 1) {
        ruleContext.attributeError("outs", "must be a singleton list");
      } else if (outs.size() == 1) {
        PathFragment soImplFilename = new PathFragment(ruleContext.getLabel().getName());
        if (LinkTargetType.DYNAMIC_LIBRARY != LinkTargetType.EXECUTABLE) {
          soImplFilename = soImplFilename.replaceName(
              "lib" + soImplFilename.getBaseName() + LinkTargetType.DYNAMIC_LIBRARY.getExtension());
        }
        soImplFilename = soImplFilename.replaceName(outs.get(0));
        if (!soImplFilename.getPathString().endsWith(".so")) { // Sanity check.
          ruleContext.attributeError("outs", "file name must end in '.so'");
        }

        if (createDynamicLibrary) {
          soImplArtifact = ruleContext.getPackageRelativeArtifact(
              soImplFilename, ruleContext.getConfiguration().getBinDirectory());
        }
      }
    }

    if (ruleContext.getRule().isAttrDefined("srcs", BuildType.LABEL_LIST)) {
      ruleContext.checkSrcsSamePackage(true);
    }
    if (ruleContext.getRule().isAttrDefined("textual_hdrs", BuildType.LABEL_LIST)) {
      helper.addPublicTextualHeaders(
          ruleContext.getPrerequisiteArtifacts("textual_hdrs", Mode.TARGET).list());
    }

    if (common.getLinkopts().contains("-static")) {
      ruleContext.attributeWarning("linkopts", "Using '-static' here won't work. "
                                   + "Did you mean to use 'linkstatic=1' instead?");
    }

    helper.setCreateDynamicLibrary(createDynamicLibrary);
    helper.setDynamicLibrary(soImplArtifact);

    // If the reason we're not creating a dynamic library is that the toolchain
    // doesn't support it, then register an action which complains when triggered,
    // which only happens when some rule explicitly depends on the dynamic library.
    if (!createDynamicLibrary && !supportsDynamicLinker) {
      Artifact solibArtifact = CppHelper.getLinkedArtifact(
          ruleContext, LinkTargetType.DYNAMIC_LIBRARY);
      ruleContext.registerAction(new FailAction(ruleContext.getActionOwner(),
          ImmutableList.of(solibArtifact), "Toolchain does not support dynamic linking"));
    } else if (!createDynamicLibrary
        && ruleContext.attributes().isConfigurable("srcs", BuildType.LABEL_LIST)) {
    // If "srcs" is configurable, the .so output is always declared because the logic that
    // determines implicit outs doesn't know which value of "srcs" will ultimately get chosen. Here,
    // where we *do* have the correct value, it may not contain any source files to generate an
    // .so with. If that's the case, register a fake generating action to prevent a "no generating
    // action for this artifact" error.
      Artifact solibArtifact = CppHelper.getLinkedArtifact(
          ruleContext, LinkTargetType.DYNAMIC_LIBRARY);
      ruleContext.registerAction(new FailAction(ruleContext.getActionOwner(),
          ImmutableList.of(solibArtifact), "configurable \"srcs\" triggers an implicit .so output "
          + "even though there are no sources to compile in this configuration"));
    }

    /*
     * Add the libraries from srcs, if any. For static/mostly static
     * linking we setup the dynamic libraries if there are no static libraries
     * to choose from. Path to the libraries will be mangled to avoid using
     * absolute path names on the -rpath, but library filenames will be
     * preserved (since some libraries might have SONAME tag) - symlink will
     * be created to the parent directory instead.
     *
     * For compatibility with existing BUILD files, any ".a" or ".lo" files listed in
     * srcs are assumed to be position-independent code, or at least suitable for
     * inclusion in shared libraries, unless they end with ".nopic.a" or ".nopic.lo".
     *
     * Note that some target platforms do not require shared library code to be PIC.
     */
    Iterable<LibraryToLink> staticLibrariesFromSrcs =
        LinkerInputs.opaqueLibrariesToLink(precompiledFiles.getStaticLibraries());
    helper.addStaticLibraries(staticLibrariesFromSrcs);
    helper.addPicStaticLibraries(Iterables.filter(staticLibrariesFromSrcs, PIC_STATIC_FILTER));
    helper.addPicStaticLibraries(precompiledFiles.getPicStaticLibraries());
    helper.addDynamicLibraries(Iterables.transform(precompiledFiles.getSharedLibraries(),
        new Function<Artifact, LibraryToLink>() {
      @Override
      public LibraryToLink apply(Artifact library) {
        return common.getDynamicLibrarySymlink(library, true);
      }
    }));
    CcLibraryHelper.Info info = helper.build();

    /*
     * We always generate a static library, even if there aren't any source files.
     * This keeps things simpler by avoiding special cases when making use of the library.
     * For example, this is needed to ensure that building a library with "bazel build"
     * will also build all of the library's "deps".
     * However, we only generate a dynamic library if there are source files.
     */
    // For now, we don't add the precompiled libraries to the files to build.
    CcLinkingOutputs linkedLibraries = info.getCcLinkingOutputsExcludingPrecompiledLibraries();

    NestedSet<Artifact> artifactsToForce =
        collectHiddenTopLevelArtifacts(ruleContext, info.getCcCompilationOutputs());

    NestedSetBuilder<Artifact> filesBuilder = NestedSetBuilder.stableOrder();
    filesBuilder.addAll(LinkerInputs.toLibraryArtifacts(linkedLibraries.getStaticLibraries()));
    filesBuilder.addAll(LinkerInputs.toLibraryArtifacts(linkedLibraries.getPicStaticLibraries()));
    filesBuilder.addAll(LinkerInputs.toNonSolibArtifacts(linkedLibraries.getDynamicLibraries()));
    filesBuilder.addAll(
        LinkerInputs.toNonSolibArtifacts(linkedLibraries.getExecutionDynamicLibraries()));

    CcLinkingOutputs linkingOutputs = info.getCcLinkingOutputs();
    warnAboutEmptyLibraries(
        ruleContext, info.getCcCompilationOutputs(), linkStatic);
    NestedSet<Artifact> filesToBuild = filesBuilder.build();

    Runfiles staticRunfiles = collectRunfiles(ruleContext,
        linkingOutputs, neverLink, addDynamicRuntimeInputArtifactsToRunfiles, true);
    Runfiles sharedRunfiles = collectRunfiles(ruleContext,
        linkingOutputs, neverLink, addDynamicRuntimeInputArtifactsToRunfiles, false);

    List<Artifact> instrumentedObjectFiles = new ArrayList<>();
    instrumentedObjectFiles.addAll(info.getCcCompilationOutputs().getObjectFiles(false));
    instrumentedObjectFiles.addAll(info.getCcCompilationOutputs().getObjectFiles(true));
    InstrumentedFilesProvider instrumentedFilesProvider =
        common.getInstrumentedFilesProvider(instrumentedObjectFiles, /*withBaselineCoverage=*/true);
    targetBuilder
        .setFilesToBuild(filesToBuild)
        .addProviders(info.getProviders())
        .addSkylarkTransitiveInfo(CcSkylarkApiProvider.NAME, new CcSkylarkApiProvider())
        .addOutputGroups(info.getOutputGroups())
        .add(InstrumentedFilesProvider.class, instrumentedFilesProvider)
        .add(RunfilesProvider.class, RunfilesProvider.withData(staticRunfiles, sharedRunfiles))
        // Remove this?
        .add(CppRunfilesProvider.class, new CppRunfilesProvider(staticRunfiles, sharedRunfiles))
        .addOutputGroup(OutputGroupProvider.HIDDEN_TOP_LEVEL, artifactsToForce);

  }

  private static NestedSet<Artifact> collectHiddenTopLevelArtifacts(
      RuleContext ruleContext, CcCompilationOutputs ccCompilationOutputs) {
    // Ensure that we build all the dependencies, otherwise users may get confused.
    NestedSetBuilder<Artifact> artifactsToForceBuilder = NestedSetBuilder.stableOrder();
    boolean isLipoCollector =
        ruleContext.getFragment(CppConfiguration.class).isLipoContextCollector();
    boolean processHeadersInDependencies =
        ruleContext.getFragment(CppConfiguration.class).processHeadersInDependencies();
    boolean usePic = CppHelper.usePic(ruleContext, false);
    artifactsToForceBuilder.addTransitive(
        ccCompilationOutputs.getFilesToCompile(
            isLipoCollector, processHeadersInDependencies, usePic));
    for (OutputGroupProvider dep :
        ruleContext.getPrerequisites("deps", Mode.TARGET, OutputGroupProvider.class)) {
      artifactsToForceBuilder.addTransitive(
          dep.getOutputGroup(OutputGroupProvider.HIDDEN_TOP_LEVEL));
    }
    return artifactsToForceBuilder.build();
  }

  /**
   * Returns the type of the generated static library.
   */
  private static LinkTargetType getStaticLinkType(RuleContext context) {
    return context.attributes().get("alwayslink", Type.BOOLEAN)
        ? LinkTargetType.ALWAYS_LINK_STATIC_LIBRARY
        : LinkTargetType.STATIC_LIBRARY;
  }

  private static void warnAboutEmptyLibraries(RuleContext ruleContext,
      CcCompilationOutputs ccCompilationOutputs,
      boolean linkstaticAttribute) {
    if (ruleContext.getFragment(CppConfiguration.class).isLipoContextCollector()) {
      // Do not signal warnings in the lipo context collector configuration. These will be duly
      // signaled in the target configuration, and there can be spurious warnings since targets in
      // the LIPO context collector configuration do not compile anything.
      return;
    }
    if (ccCompilationOutputs.getObjectFiles(false).isEmpty()
        && ccCompilationOutputs.getObjectFiles(true).isEmpty()) {
      if (!linkstaticAttribute && appearsToHaveObjectFiles(ruleContext.attributes())) {
        ruleContext.attributeWarning("linkstatic",
            "setting 'linkstatic=1' is recommended if there are no object files");
      }
    } else {
      if (!linkstaticAttribute && !appearsToHaveObjectFiles(ruleContext.attributes())) {
        Artifact element = Iterables.getFirst(
            ccCompilationOutputs.getObjectFiles(false),
            ccCompilationOutputs.getObjectFiles(true).get(0));
        ruleContext.attributeWarning("srcs",
             "this library appears at first glance to have no object files, "
             + "but on closer inspection it does have something to link, e.g. "
             + element.prettyPrint() + ". "
             + "(You may have used some very confusing rule names in srcs? "
             + "Or the library consists entirely of a linker script?) "
             + "Bazel assumed linkstatic=1, but this may be inappropriate. "
             + "You may need to add an explicit '.cc' file to 'srcs'. "
             + "Alternatively, add 'linkstatic=1' to suppress this warning");
      }
    }
  }

  /**
   * Returns true if the rule (which must be a cc_library rule) appears to have object files.
   * This only looks at the rule itself, not at any other rules (from this package or other
   * packages) that it might reference.
   *
   * <p>In some cases, this may return "true" even though the rule actually has no object files.
   * For example, it will return true for a rule such as
   * <code>cc_library(name = 'foo', srcs = [':bar'])</code> because we can't tell what ':bar' is;
   * it might be a genrule that generates a source file, or it might be a genrule that generates a
   * header file. Likewise,
   * <code>cc_library(name = 'foo', srcs = select({':a': ['foo.cc'], ':b': []}))</code> returns
   * "true" even though the sources *may* be empty. This reflects the fact that there's no way
   * to tell which value "srcs" will take without knowing the rule's configuration.
   *
   * <p>In other cases, this may return "false" even though the rule actually does have object
   * files. For example, it will return false for a rule such as
   * <code>cc_library(name = 'foo', srcs = ['bar.h'])</code> but as in the other example above,
   * we can't tell whether 'bar.h' is a file name or a rule name, and 'bar.h' could in fact be the
   * name of a genrule that generates a source file.
   */
  public static boolean appearsToHaveObjectFiles(AttributeMap rule) {
    if ((rule instanceof RawAttributeMapper) && rule.isConfigurable("srcs", BuildType.LABEL_LIST)) {
      // Since this method gets called by loading phase logic (e.g. the cc_library implicit outputs
      // function), the attribute mapper may not be able to resolve configurable attributes. When
      // that's the case, there's no way to know which value a configurable "srcs" will take, so
      // we conservatively assume object files are possible.
      return true;
    }

    List<Label> srcs = rule.get("srcs", BuildType.LABEL_LIST);
    if (srcs != null) {
      for (Label srcfile : srcs) {
        /*
         * We cheat a little bit here by looking at the file extension
         * of the Label treated as file name.  In general that might
         * not necessarily work, because of the possibility that the
         * user might give a rule a funky name ending in one of these
         * extensions, e.g.
         *    genrule(name = 'foo.h', outs = ['foo.cc'], ...) // Funky rule name!
         *    cc_library(name = 'bar', srcs = ['foo.h']) // This DOES have object files.
         */
        if (!NO_OBJECT_GENERATING_FILETYPES.matches(srcfile.getName())) {
          return true;
        }
      }
    }
    return false;
  }
}
