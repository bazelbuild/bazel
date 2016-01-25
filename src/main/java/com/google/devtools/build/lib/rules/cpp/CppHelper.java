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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MiddlemanFactory;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.rules.cpp.CcLinkParams.Linkstamp;
import com.google.devtools.build.lib.rules.cpp.CppCompilationContext.Builder;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.IncludeScanningUtil;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.LipoMode;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * Helper class for functionality shared by cpp related rules.
 *
 * <p>This class can be used only after the loading phase.
 */
public class CppHelper {
  // TODO(bazel-team): should this use Link.SHARED_LIBRARY_FILETYPES?
  public static final FileTypeSet SHARED_LIBRARY_FILETYPES = FileTypeSet.of(
      CppFileTypes.SHARED_LIBRARY,
      CppFileTypes.VERSIONED_SHARED_LIBRARY);

  private static final FileTypeSet CPP_FILETYPES = FileTypeSet.of(
      CppFileTypes.CPP_HEADER,
      CppFileTypes.CPP_SOURCE);

  private static final ImmutableList<String> LINKOPTS_PREREQUISITE_LABEL_KINDS =
      ImmutableList.of("deps", "srcs");

  private CppHelper() {
    // prevents construction
  }

  /**
   * Merges the STL and toolchain contexts into context builder. The STL is automatically determined
   * using the ":stl" attribute.
   */
  public static void mergeToolchainDependentContext(RuleContext ruleContext,
      Builder contextBuilder) {
    if (ruleContext.getRule().getAttributeDefinition(":stl") != null) {
      TransitiveInfoCollection stl = ruleContext.getPrerequisite(":stl", Mode.TARGET);
      if (stl != null) {
        // TODO(bazel-team): Clean this up.
        contextBuilder.addSystemIncludeDir(
            stl.getLabel().getPackageIdentifier().getPathFragment().getRelative("gcc3"));
        contextBuilder.mergeDependentContext(stl.getProvider(CppCompilationContext.class));
      }
    }
    CcToolchainProvider toolchain = getToolchain(ruleContext);
    if (toolchain != null) {
      contextBuilder.mergeDependentContext(toolchain.getCppCompilationContext());
    }
  }

  /**
   * Returns the malloc implementation for the given target.
   */
  public static TransitiveInfoCollection mallocForTarget(RuleContext ruleContext) {
    if (ruleContext.getFragment(CppConfiguration.class).customMalloc() != null) {
      return ruleContext.getPrerequisite(":default_malloc", Mode.TARGET);
    } else {
      return ruleContext.getPrerequisite("malloc", Mode.TARGET);
    }
  }

  /**
   * Expands Make variables in a list of string and tokenizes the result. If the package feature
   * no_copts_tokenization is set, tokenize only items consisting of a single make variable.
   *
   * @param ruleContext the ruleContext to be used as the context of Make variable expansion
   * @param attributeName the name of the attribute to use in error reporting
   * @param input the list of strings to expand
   * @return a list of strings containing the expanded and tokenized values for the
   *         attribute
   */
  // TODO(bazel-team): Move to CcCommon; refactor CcPlugin to use either CcLibraryHelper or
  // CcCommon.
  static List<String> expandMakeVariables(
      RuleContext ruleContext, String attributeName, List<String> input) {
    boolean tokenization =
        !ruleContext.getFeatures().contains("no_copts_tokenization");

    List<String> tokens = new ArrayList<>();
    for (String token : input) {
      try {
        // Legacy behavior: tokenize all items.
        if (tokenization) {
          ruleContext.tokenizeAndExpandMakeVars(tokens, attributeName, token);
        } else {
          String exp = ruleContext.expandSingleMakeVariable(attributeName, token);
          if (exp != null) {
            ShellUtils.tokenize(tokens, exp);
          } else {
            tokens.add(ruleContext.expandMakeVariables(attributeName, token));
          }
        }
      } catch (ShellUtils.TokenizationException e) {
        ruleContext.attributeError(attributeName, e.getMessage());
      }
    }
    return ImmutableList.copyOf(tokens);
  }

  /**
   * Appends the tokenized values of the copts attribute to copts.
   */
  public static ImmutableList<String> getAttributeCopts(RuleContext ruleContext, String attr) {
    Preconditions.checkArgument(ruleContext.getRule().isAttrDefined(attr, Type.STRING_LIST));
    List<String> unexpanded = ruleContext.attributes().get(attr, Type.STRING_LIST);

    return ImmutableList.copyOf(expandMakeVariables(ruleContext, attr, unexpanded));
  }

  /**
   * Expands attribute value either using label expansion
   * (if attemptLabelExpansion == {@code true} and it does not look like make
   * variable or flag) or tokenizes and expands make variables.
   */
  public static void expandAttribute(RuleContext ruleContext,
      List<String> values, String attrName, String attrValue, boolean attemptLabelExpansion) {
    if (attemptLabelExpansion && CppHelper.isLinkoptLabel(attrValue)) {
      if (!CppHelper.expandLabel(ruleContext, values, attrValue)) {
        ruleContext.attributeError(attrName, "could not resolve label '" + attrValue + "'");
      }
    } else {
      ruleContext.tokenizeAndExpandMakeVars(values, attrName, attrValue);
    }
  }

  /**
   * Determines if a linkopt can be a label. Linkopts come in 2 varieties:
   * literals -- flags like -Xl and makefile vars like $(LD) -- and labels,
   * which we should expand into filenames.
   *
   * @param linkopt the link option to test.
   * @return true if the linkopt is not a flag (starting with "-") or a makefile
   *         variable (starting with "$");
   */
  private static boolean isLinkoptLabel(String linkopt) {
    return !linkopt.startsWith("$") && !linkopt.startsWith("-");
  }

  /**
   * Expands a label against the target's deps, adding the expanded path strings
   * to the linkopts.
   *
   * @param linkopts the linkopts to add the expanded label to
   * @param labelName the name of the label to expand
   * @return true if the label was expanded successfully, false otherwise
   */
  private static boolean expandLabel(RuleContext ruleContext, List<String> linkopts,
      String labelName) {
    try {
      Label label = ruleContext.getLabel().getRelative(labelName);
      for (String prereqKind : LINKOPTS_PREREQUISITE_LABEL_KINDS) {
        for (FileProvider target : ruleContext
            .getPrerequisites(prereqKind, Mode.TARGET, FileProvider.class)) {
          if (target.getLabel().equals(label)) {
            for (Artifact artifact : target.getFilesToBuild()) {
              linkopts.add(artifact.getExecPathString());
            }
            return true;
          }
        }
      }
    } catch (LabelSyntaxException e) {
      // Quietly ignore and fall through.
    }
    linkopts.add(labelName);
    return false;
  }

  /**
   * This almost trivial method looks up the :cc_toolchain attribute on the rule context, makes sure
   * that it refers to a rule that has a {@link CcToolchainProvider} (gives an error otherwise), and
   * returns a reference to that {@link CcToolchainProvider}. The method only returns {@code null}
   * if there is no such attribute (this is currently not an error).
   */
  @Nullable public static CcToolchainProvider getToolchain(RuleContext ruleContext) {
    if (ruleContext.attributes().getAttributeDefinition(":cc_toolchain") == null) {
      // TODO(bazel-team): Report an error or throw an exception in this case.
      return null;
    }
    TransitiveInfoCollection dep = ruleContext.getPrerequisite(":cc_toolchain", Mode.TARGET);
    return getToolchain(ruleContext, dep);
  }

  /**
   * This almost trivial method makes sure that the given info collection has a {@link
   * CcToolchainProvider} (gives an error otherwise), and returns a reference to that {@link
   * CcToolchainProvider}. The method never returns {@code null}, even if there is no toolchain.
   */
  public static CcToolchainProvider getToolchain(RuleContext ruleContext,
      TransitiveInfoCollection dep) {
    // TODO(bazel-team): Consider checking this generally at the attribute level.
    if ((dep == null) || (dep.getProvider(CcToolchainProvider.class) == null)) {
      ruleContext.ruleError("The selected C++ toolchain is not a cc_toolchain rule");
      return CcToolchainProvider.EMPTY_TOOLCHAIN_IS_ERROR;
    }
    return dep.getProvider(CcToolchainProvider.class);
  }

  /**
   * Returns the directory where object files are created.
   */
  public static PathFragment getObjDirectory(Label ruleLabel) {
    return AnalysisUtils.getUniqueDirectory(ruleLabel, new PathFragment("_objs"));
  }

  /**
   * Creates a grep-includes ExtractInclusions action for generated sources/headers in the
   * needsIncludeScanning() BuildConfiguration case. Returns a map from original header
   * Artifact to the output Artifact of grepping over it. The return value only includes
   * entries for generated sources or headers when --extract_generated_inclusions is enabled.
   *
   * <p>Previously, incremental rebuilds redid all include scanning work
   * for a given .cc source in serial. For high-latency file systems, this could cause
   * performance problems if many headers are generated.
   */
  @Nullable
  public static final Map<Artifact, Artifact> createExtractInclusions(
      RuleContext ruleContext, CppSemantics semantics, Iterable<Artifact> prerequisites) {
    Map<Artifact, Artifact> extractions = new HashMap<>();
    for (Artifact prerequisite : prerequisites) {
      if (extractions.containsKey(prerequisite)) {
        // Don't create duplicate actions just because user specified same header file twice.
        continue;
      }
      Artifact scanned = createExtractInclusions(ruleContext, semantics, prerequisite);
      if (scanned != null) {
        extractions.put(prerequisite, scanned);
      }
    }
    return extractions;
  }

  /**
   * Creates a grep-includes ExtractInclusions action for generated  sources/headers in the
   * needsIncludeScanning() BuildConfiguration case.
   *
   * <p>Previously, incremental rebuilds redid all include scanning work for a given
   * .cc source in serial. For high-latency file systems, this could cause
   * performance problems if many headers are generated.
   */
  private static final Artifact createExtractInclusions(
      RuleContext ruleContext, CppSemantics semantics, Artifact prerequisite) {
    if (ruleContext != null
        && semantics.needsIncludeScanning(ruleContext)
        && !prerequisite.isSourceArtifact()
        && CPP_FILETYPES.matches(prerequisite.getFilename())) {
      Artifact scanned = getIncludesOutput(ruleContext, semantics, prerequisite);
      ruleContext.registerAction(
          new ExtractInclusionAction(ruleContext.getActionOwner(), prerequisite, scanned));
      return scanned;
    }
    return null;
  }

  private static Artifact getIncludesOutput(
      RuleContext ruleContext, CppSemantics semantics, Artifact src) {
    Root root = semantics.getGreppedIncludesDirectory(ruleContext);
    PathFragment relOut = IncludeScanningUtil.getRootRelativeOutputPath(src.getExecPath());
    return ruleContext.getShareableArtifact(relOut, root);
  }

  /**
   * Returns the linked artifact.
   */
  public static Artifact getLinkedArtifact(RuleContext ruleContext, LinkTargetType linkType) {
    PathFragment name = new PathFragment(ruleContext.getLabel().getName());
    if (linkType != LinkTargetType.EXECUTABLE) {
      name = name.replaceName("lib" + name.getBaseName() + linkType.getExtension());
    }

    return ruleContext.getPackageRelativeArtifact(
        name, ruleContext.getConfiguration().getBinDirectory());
  }

  /**
   * Resolves the linkstamp collection from the {@code CcLinkParams} into a map.
   *
   * <p>Emits a warning on the rule if there are identical linkstamp artifacts with different
   * compilation contexts.
   */
  public static Map<Artifact, ImmutableList<Artifact>> resolveLinkstamps(RuleErrorConsumer listener,
      CcLinkParams linkParams) {
    Map<Artifact, ImmutableList<Artifact>> result = new LinkedHashMap<>();
    for (Linkstamp pair : linkParams.getLinkstamps()) {
      Artifact artifact = pair.getArtifact();
      if (result.containsKey(artifact)) {
        listener.ruleWarning("rule inherits the '" + artifact.toDetailString()
            + "' linkstamp file from more than one cc_library rule");
      }
      result.put(artifact, pair.getDeclaredIncludeSrcs());
    }
    return result;
  }

  public static void addTransitiveLipoInfoForCommonAttributes(
      RuleContext ruleContext,
      CcCompilationOutputs outputs,
      NestedSetBuilder<IncludeScannable> scannableBuilder) {

    TransitiveLipoInfoProvider stl = null;
    if (ruleContext.getRule().getAttributeDefinition(":stl") != null &&
        ruleContext.getPrerequisite(":stl", Mode.TARGET) != null) {
      // If the attribute is defined, it is never null.
      stl = ruleContext.getPrerequisite(":stl", Mode.TARGET)
          .getProvider(TransitiveLipoInfoProvider.class);
    }
    if (stl != null) {
      scannableBuilder.addTransitive(stl.getTransitiveIncludeScannables());
    }

    for (TransitiveLipoInfoProvider dep :
        ruleContext.getPrerequisites("deps", Mode.TARGET, TransitiveLipoInfoProvider.class)) {
      scannableBuilder.addTransitive(dep.getTransitiveIncludeScannables());
    }

    if (ruleContext.attributes().has("malloc", BuildType.LABEL)) {
      TransitiveInfoCollection malloc = mallocForTarget(ruleContext);
      TransitiveLipoInfoProvider provider = malloc.getProvider(TransitiveLipoInfoProvider.class);
      if (provider != null) {
        scannableBuilder.addTransitive(provider.getTransitiveIncludeScannables());
      }
    }

    for (IncludeScannable scannable : outputs.getLipoScannables()) {
      Preconditions.checkState(scannable.getIncludeScannerSources().size() == 1);
      scannableBuilder.add(scannable);
    }
  }

  // TODO(bazel-team): figure out a way to merge these 2 methods. See the Todo in
  // CcCommonConfiguredTarget.noCoptsMatches().
  /**
   * Determines if we should apply -fPIC for this rule's C++ compilations. This determination
   * is generally made by the global C++ configuration settings "needsPic" and
   * and "usePicForBinaries". However, an individual rule may override these settings by applying
   * -fPIC" to its "nocopts" attribute. This allows incompatible rules to "opt out" of global PIC
   * settings (see bug: "Provide a way to turn off -fPIC for targets that can't be built that way").
   *
   * @param ruleContext the context of the rule to check
   * @param forBinary true if compiling for a binary, false if for a shared library
   * @return true if this rule's compilations should apply -fPIC, false otherwise
   */
  public static boolean usePic(RuleContext ruleContext, boolean forBinary) {
    if (CcCommon.noCoptsMatches("-fPIC", ruleContext)) {
      return false;
    }
    CppConfiguration config = ruleContext.getFragment(CppConfiguration.class);
    return forBinary ? config.usePicObjectsForBinaries() : config.needsPic();
  }

  /**
   * Returns the LIPO context provider for configured target,
   * or null if such a provider doesn't exist.
   */
  public static LipoContextProvider getLipoContextProvider(RuleContext ruleContext) {
    if (ruleContext.getRule().getAttributeDefinition(":lipo_context_collector") == null) {
      return null;
    }

    TransitiveInfoCollection dep =
        ruleContext.getPrerequisite(":lipo_context_collector", Mode.DONT_CHECK);
    return (dep != null) ? dep.getProvider(LipoContextProvider.class) : null;
  }

  // Creates CppModuleMap object, and adds it to C++ compilation context.
  public static CppModuleMap addCppModuleMapToContext(RuleContext ruleContext,
      CppCompilationContext.Builder contextBuilder) {
    // Create the module map artifact as a genfile.
    Artifact mapFile = ruleContext.getPackageRelativeArtifact(
        ruleContext.getLabel().getName()
            + Iterables.getOnlyElement(CppFileTypes.CPP_MODULE_MAP.getExtensions()),
        ruleContext.getConfiguration().getGenfilesDirectory());    CppModuleMap moduleMap =
        new CppModuleMap(mapFile, ruleContext.getLabel().toString());
    contextBuilder.setCppModuleMap(moduleMap);
    return moduleMap;
  }

  /**
   * Returns a middleman for all files to build for the given configured target,
   * substituting shared library artifacts with corresponding solib symlinks. If
   * multiple calls are made, then it returns the same artifact for configurations
   * with the same internal directory.
   *
   * <p>The resulting middleman only aggregates the inputs and must be expanded
   * before populating the set of files necessary to execute an action.
   */
  static List<Artifact> getAggregatingMiddlemanForCppRuntimes(RuleContext ruleContext,
      String purpose, TransitiveInfoCollection dep, String solibDirOverride,
      BuildConfiguration configuration) {
    return getMiddlemanInternal(
        ruleContext.getAnalysisEnvironment(), ruleContext, ruleContext.getActionOwner(), purpose,
        dep, true, true, solibDirOverride, configuration);
  }

  @VisibleForTesting
  public static List<Artifact> getAggregatingMiddlemanForTesting(AnalysisEnvironment env,
      RuleContext ruleContext, ActionOwner owner, String purpose, TransitiveInfoCollection dep,
      boolean useSolibSymlinks, BuildConfiguration configuration) {
    return getMiddlemanInternal(
        env, ruleContext, owner, purpose, dep, useSolibSymlinks, false, null, configuration);
  }

  /**
   * Internal implementation for getAggregatingMiddlemanForCppRuntimes.
   */
  private static List<Artifact> getMiddlemanInternal(AnalysisEnvironment env,
      RuleContext ruleContext, ActionOwner actionOwner, String purpose,
      TransitiveInfoCollection dep, boolean useSolibSymlinks, boolean isCppRuntime,
      String solibDirOverride, BuildConfiguration configuration) {
    if (dep == null) {
      return ImmutableList.of();
    }
    MiddlemanFactory factory = env.getMiddlemanFactory();
    Iterable<Artifact> artifacts = dep.getProvider(FileProvider.class).getFilesToBuild();
    if (useSolibSymlinks) {
      List<Artifact> symlinkedArtifacts = new ArrayList<>();
      for (Artifact artifact : artifacts) {
        symlinkedArtifacts.add(solibArtifactMaybe(
            ruleContext, artifact, isCppRuntime, solibDirOverride, configuration));
      }
      artifacts = symlinkedArtifacts;
      purpose += "_with_solib";
    }
    return ImmutableList.of(
        factory.createMiddlemanAllowMultiple(env, actionOwner, ruleContext.getPackageDirectory(),
            purpose, artifacts, configuration.getMiddlemanDirectory()));
  }

  /**
   * If the artifact is a shared library, returns the solib symlink artifact associated with it.
   *
   * @param ruleContext the context of the rule that creates the symlink
   * @param artifact the library the solib symlink should point to
   * @param isCppRuntime whether the library is a C++ runtime
   * @param solibDirOverride if not null, forces the solib symlink to be in this directory
   */
  private static Artifact solibArtifactMaybe(RuleContext ruleContext, Artifact artifact,
      boolean isCppRuntime, String solibDirOverride, BuildConfiguration configuration) {
    if (SHARED_LIBRARY_FILETYPES.matches(artifact.getFilename())) {
      return isCppRuntime
        ? SolibSymlinkAction.getCppRuntimeSymlink(
            ruleContext, artifact, solibDirOverride, configuration)
            .getArtifact()
        : SolibSymlinkAction.getDynamicLibrarySymlink(
            ruleContext, artifact, false, true, configuration)
            .getArtifact();
    } else {
      return artifact;
    }
  }

  /**
   * Returns the type of archives being used.
   */
  public static Link.ArchiveType archiveType(BuildConfiguration config) {
    CppConfiguration cppConfig = config.getFragment(CppConfiguration.class);
    return cppConfig.archiveType();
  }

  /**
   * Returns the FDO build subtype.
   */
  public static String getFdoBuildStamp(CppConfiguration cppConfiguration) {
    if (cppConfiguration.getFdoSupport().isAutoFdoEnabled()) {
      return (cppConfiguration.getLipoMode() == LipoMode.BINARY) ? "ALIPO" : "AFDO";
    }
    if (cppConfiguration.isFdo()) {
      return (cppConfiguration.getLipoMode() == LipoMode.BINARY) ? "LIPO" : "FDO";
    }
    return null;
  }

  /**
   * Returns a relative path to the bin directory for data in AutoFDO LIPO mode.
   */
  public static PathFragment getLipoDataBinFragment(BuildConfiguration configuration) {
    PathFragment parent = configuration.getBinFragment().getParentDirectory();
    return parent.replaceName(parent.getBaseName() + "-lipodata")
        .getChild(configuration.getBinFragment().getBaseName());
  }

  /**
   * Returns a relative path to the genfiles directory for data in AutoFDO LIPO mode.
   */
  public static PathFragment getLipoDataGenfilesFragment(BuildConfiguration configuration) {
    PathFragment parent = configuration.getGenfilesFragment().getParentDirectory();
    return parent.replaceName(parent.getBaseName() + "-lipodata")
        .getChild(configuration.getGenfilesFragment().getBaseName());
  }

  /**
   * Creates an action to strip an executable.
   */
  public static void createStripAction(RuleContext context,
      CppConfiguration cppConfiguration, Artifact input, Artifact output) {
    context.registerAction(new SpawnAction.Builder()
        .addInput(input)
        .addTransitiveInputs(CppHelper.getToolchain(context).getStrip())
        .addOutput(output)
        .useDefaultShellEnvironment()
        .setExecutable(cppConfiguration.getStripExecutable())
        .addArguments("-S", "-p", "-o", output.getExecPathString())
        .addArguments("-R", ".gnu.switches.text.quote_paths")
        .addArguments("-R", ".gnu.switches.text.bracket_paths")
        .addArguments("-R", ".gnu.switches.text.system_paths")
        .addArguments("-R", ".gnu.switches.text.cpp_defines")
        .addArguments("-R", ".gnu.switches.text.cpp_includes")
        .addArguments("-R", ".gnu.switches.text.cl_args")
        .addArguments("-R", ".gnu.switches.text.lipo_info")
        .addArguments("-R", ".gnu.switches.text.annotation")
        .addArguments(cppConfiguration.getStripOpts())
        .addArgument(input.getExecPathString())
        .setProgressMessage("Stripping " + output.prettyPrint() + " for " + context.getLabel())
        .setMnemonic("CcStrip")
        .build(context));
  }
}
