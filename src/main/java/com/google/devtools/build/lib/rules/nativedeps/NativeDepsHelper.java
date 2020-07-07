// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.nativedeps;

import static com.google.devtools.build.lib.rules.cpp.CppRuleClasses.STATIC_LINKING_MODE;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.rules.cpp.CcCommon;
import com.google.devtools.build.lib.rules.cpp.CcCompilationOutputs;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext.Linkstamp;
import com.google.devtools.build.lib.rules.cpp.CcLinkingHelper;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppBuildInfo;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppHelper;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.rules.cpp.FdoContext;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
import com.google.devtools.build.lib.rules.cpp.Link;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.LinkingMode;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;

/**
 * Helper class to create a dynamic library for rules which support integration with native code.
 *
 * <p>This library gets created by the build system by linking all C++ libraries in the transitive
 * closure of the dependencies into a standalone dynamic library, with some exceptions. It usually
 * does not include neverlink libraries or C++ binaries (or their transitive dependencies). Note
 * that some rules are implicitly neverlink.
 */
public abstract class NativeDepsHelper {
  /**
   * An implementation of {@link
   * com.google.devtools.build.lib.rules.cpp.CppLinkAction.LinkArtifactFactory} that can create
   * artifacts anywhere.
   *
   * <p>Necessary because the actions of nativedeps libraries should be shareable, and thus cannot
   * be under the package directory.
   */
  private static final CppLinkAction.LinkArtifactFactory SHAREABLE_LINK_ARTIFACT_FACTORY =
      new CppLinkAction.LinkArtifactFactory() {
        @Override
        public Artifact create(
            ActionConstructionContext actionConstructionContext,
            RepositoryName repositoryName,
            BuildConfiguration configuration,
            PathFragment rootRelativePath) {
          return actionConstructionContext.getShareableArtifact(
              rootRelativePath, configuration.getBinDirectory(repositoryName));
        }
      };

  private NativeDepsHelper() {}

  private static final String ANDROID_UNIQUE_DIR = "nativedeps";

  /**
   * Creates an Action to create a dynamic library for Android by linking all native code (C/C++)
   * libraries in the transitive dependency closure of a rule.
   *
   * <p>We link the native deps in the equivalent of linkstatic=1, linkshared=1 mode.
   *
   * <p>linkstatic=1 means mostly-static mode, i.e. we select the ".a" (or ".pic.a") files, but we
   * don't include "-static" in linkopts.
   *
   * <p>linkshared=1 means we prefer the ".pic.a" files to the ".a" files, and the LinkTargetType is
   * set to DYNAMIC_LIBRARY which causes Link.java to include "-shared" in the linker options.
   *
   * <p>It is possible that this function may have no work to do if there are no native libraries in
   * the transitive closure, or if the only native libraries in the transitive closure are already
   * shared libraries. In this case, this function returns {@code null}.
   *
   * @param ruleContext the rule context to determine the native deps library
   * @param ccInfo the {@link CcInfo} for the rule, collected with linkstatic = 1 and linkshared = 1
   * @param cppSemantics to use for linkstamp compiles
   * @return the native deps library, or null if there was no code which needed to be linked in the
   *     transitive closure.
   */
  public static Artifact linkAndroidNativeDepsIfPresent(
      final RuleContext ruleContext,
      CcInfo ccInfo,
      final BuildConfiguration configuration,
      CcToolchainProvider toolchain,
      CppSemantics cppSemantics)
      throws InterruptedException, RuleErrorException {
    if (!containsCodeToLink(ccInfo.getCcLinkingContext().getLibraries())) {
      return null;
    }

    PathFragment labelName = PathFragment.create(ruleContext.getLabel().getName());
    Artifact nativeDeps = ruleContext.getUniqueDirectoryArtifact(ANDROID_UNIQUE_DIR,
        labelName.replaceName("lib" + labelName.getBaseName() + ".so"),
        configuration.getBinDirectory(ruleContext.getRule().getRepository()));

    return createNativeDepsAction(
            ruleContext,
            ccInfo,
            /* extraLinkOpts= */ ImmutableList.of(),
            configuration,
            toolchain,
            nativeDeps,
            configuration.getBinDirectory(ruleContext.getRule().getRepository()),
            cppSemantics)
        .getLibrary();
  }

  /** Determines if there is any code to be linked in the input iterable. */
  private static boolean containsCodeToLink(NestedSet<LibraryToLink> libraries) {
    for (LibraryToLink library : libraries.toList()) {
      if (containsCodeToLink(library)) {
        return true;
      }
    }
    return false;
  }

  /** Determines if the input library is or contains an archive which must be linked. */
  private static boolean containsCodeToLink(LibraryToLink library) {
    if (library.getStaticLibrary() == null && library.getPicStaticLibrary() == null) {
      // this is a shared library so we're going to have to copy it
      return false;
    }
    Iterable<Artifact> objectFiles;
    if (library.getObjectFiles() != null) {
      objectFiles = library.getObjectFiles();
    } else if (library.getPicObjectFiles() != null) {
      objectFiles = library.getPicObjectFiles();
    } else {
      // this is an opaque library so we're going to have to link it
      return true;
    }
    for (Artifact object : objectFiles) {
      if (!Link.SHARED_LIBRARY_FILETYPES.matches(object.getFilename())) {
        // this library was built with a non-shared-library object so we should link it
        return true;
      }
    }
    // there weren't any artifacts besides shared libraries compiled in the library
    return false;
  }

  public static NativeDepsRunfiles createNativeDepsAction(
      final RuleContext ruleContext,
      CcInfo ccInfo,
      Collection<String> extraLinkOpts,
      BuildConfiguration configuration,
      CcToolchainProvider toolchain,
      Artifact nativeDeps,
      ArtifactRoot bindirIfShared,
      CppSemantics cppSemantics)
      throws InterruptedException, RuleErrorException {
    CcLinkingContext ccLinkingContext = ccInfo.getCcLinkingContext();
    Preconditions.checkState(
        ruleContext.isLegalFragment(CppConfiguration.class),
        "%s does not have access to CppConfiguration",
        ruleContext.getRule().getRuleClass());
    List<String> linkopts = new ArrayList<>(extraLinkOpts);
    linkopts.addAll(ccLinkingContext.getFlattenedUserLinkFlags());

    CppHelper.checkLinkstampsUnique(ruleContext, ccLinkingContext.getLinkstamps().toList());
    ImmutableSet<Linkstamp> linkstamps = ccLinkingContext.getLinkstamps().toSet();
    List<Artifact> buildInfoArtifacts =
        linkstamps.isEmpty()
            ? ImmutableList.<Artifact>of()
            : ruleContext
                .getAnalysisEnvironment()
                .getBuildInfo(
                    AnalysisUtils.isStampingEnabled(ruleContext, configuration),
                    CppBuildInfo.KEY,
                    configuration);

    ImmutableSet.Builder<String> requestedFeatures =
        ImmutableSet.<String>builder().addAll(ruleContext.getFeatures()).add(STATIC_LINKING_MODE);
    if (!ruleContext.getDisabledFeatures().contains(CppRuleClasses.LEGACY_WHOLE_ARCHIVE)) {
      requestedFeatures.add(CppRuleClasses.LEGACY_WHOLE_ARCHIVE);
    }
    FeatureConfiguration featureConfiguration =
        CcCommon.configureFeaturesOrReportRuleError(
            ruleContext,
            /* requestedFeatures= */ requestedFeatures.build(),
            /* unsupportedFeatures= */ ruleContext.getDisabledFeatures(),
            toolchain,
            cppSemantics);

    boolean shareNativeDeps = configuration.getFragment(CppConfiguration.class).shareNativeDeps();
    boolean isThinLtoDisabledOnlyForLinkStaticTestAndTestOnlyTargets =
        !featureConfiguration.isEnabled(
                CppRuleClasses.THIN_LTO_ALL_LINKSTATIC_USE_SHARED_NONLTO_BACKENDS)
            && featureConfiguration.isEnabled(
                CppRuleClasses.THIN_LTO_LINKSTATIC_TESTS_USE_SHARED_NONLTO_BACKENDS);
    boolean isTestOrTestOnlyTarget = ruleContext.isTestOnlyTarget() || ruleContext.isTestTarget();
    Artifact sharedLibrary;
    if (shareNativeDeps) {
      PathFragment sharedPath =
          getSharedNativeDepsPath(
              ccLinkingContext.getStaticModeParamsForDynamicLibraryLibraries(),
              linkopts,
              linkstamps.stream()
                  .map(CcLinkingContext.Linkstamp::getArtifact)
                  .collect(ImmutableList.toImmutableList()),
              buildInfoArtifacts,
              ruleContext.getFeatures(),
              isTestOrTestOnlyTarget && isThinLtoDisabledOnlyForLinkStaticTestAndTestOnlyTargets);

      sharedLibrary = ruleContext.getShareableArtifact(
          sharedPath.replaceName(sharedPath.getBaseName() + ".so"),
          configuration.getBinDirectory(ruleContext.getRule().getRepository()));
    } else {
      sharedLibrary = nativeDeps;
    }
    FdoContext fdoContext = toolchain.getFdoContext();

    new CcLinkingHelper(
            ruleContext,
            ruleContext.getLabel(),
            ruleContext,
            ruleContext,
            cppSemantics,
            featureConfiguration,
            toolchain,
            fdoContext,
            configuration,
            ruleContext.getFragment(CppConfiguration.class),
            ruleContext.getSymbolGenerator(),
            TargetUtils.getExecutionInfo(
                ruleContext.getRule(), ruleContext.isAllowTagsPropagation()))
        .setGrepIncludes(CppHelper.getGrepIncludes(ruleContext))
        .setIsStampingEnabled(AnalysisUtils.isStampingEnabled(ruleContext))
        .setTestOrTestOnlyTarget(ruleContext.isTestTarget() || ruleContext.isTestOnlyTarget())
        .setLinkerOutputArtifact(sharedLibrary)
        .setLinkingMode(LinkingMode.STATIC)
        .addLinkopts(extraLinkOpts)
        .setNativeDeps(true)
        .setNeverLink(true)
        .setShouldCreateStaticLibraries(false)
        .addCcLinkingContexts(ImmutableList.of(ccLinkingContext))
        .setLinkArtifactFactory(SHAREABLE_LINK_ARTIFACT_FACTORY)
        .setDynamicLinkType(LinkTargetType.DYNAMIC_LIBRARY)
        .link(CcCompilationOutputs.EMPTY);

    if (shareNativeDeps) {
      ruleContext.registerAction(
          SymlinkAction.toArtifact(ruleContext.getActionOwner(), sharedLibrary, nativeDeps, null));
      return new NativeDepsRunfiles(nativeDeps);
    }

    return new NativeDepsRunfiles(sharedLibrary);
  }

  /**
   * Returns the path of the shared native library. The name must be generated based on the
   * rule-specific inputs to the link actions. At this point this includes order-sensitive list of
   * linker inputs and options collected from the transitive closure and linkstamp-related artifacts
   * that are compiled during linking. All those inputs can be affected by modifying target
   * attributes (srcs/deps/stamp/etc). However, target build configuration can be ignored since it
   * will either change output directory (in case of different configuration instances) or will not
   * affect anything (if two targets use same configuration). Final goal is for all native libraries
   * that use identical linker command to use same output name.
   *
   * <p>TODO(bazel-team): (2010) Currently process of identifying parameters that can affect native
   * library name is manual and should be kept in sync with the code in the
   * CppLinkAction.Builder/CppLinkAction/Link classes which are responsible for generating linker
   * command line. Ideally we should reuse generated command line for both purposes - selecting a
   * name of the native library and using it as link action payload. For now, correctness of the
   * method below is only ensured by validations in the CppLinkAction.Builder.build() method.
   */
  private static PathFragment getSharedNativeDepsPath(
      Iterable<Artifact> linkerInputs,
      Collection<String> linkopts,
      Iterable<Artifact> linkstamps,
      Iterable<Artifact> buildInfoArtifacts,
      Collection<String> features,
      boolean isTestTargetInPartiallyDisabledThinLtoCase) {
    Fingerprint fp = new Fingerprint();
    int linkerInputsSize = 0;
    for (Artifact input : linkerInputs) {
      fp.addString(input.getExecPathString());
      linkerInputsSize++;
    }
    fp.addStrings(linkopts);
    int linkstampsSize = 0;
    for (Artifact input : linkstamps) {
      fp.addString(input.getExecPathString());
      linkstampsSize++;
    }
    // TODO(b/120206809): remove debugging info here (and in this whole filename construction).
    String linkstampsString = Integer.toString(linkstampsSize);
    if (linkstampsSize > 1) {
      Set<Artifact> identitySet = Sets.newIdentityHashSet();
      Iterables.addAll(identitySet, linkstamps);
      if (identitySet.size() < linkstampsSize) {
        linkstampsString += "_" + identitySet.size();
      }
      ImmutableSet<Artifact> uniqueLinkStamps = ImmutableSet.copyOf(linkstamps);
      if (uniqueLinkStamps.size() < linkstampsSize) {
        linkstampsString += "__" + uniqueLinkStamps.size();
      }
    }
    int buildInfoSize = 0;
    for (Artifact input : buildInfoArtifacts) {
      fp.addString(input.getExecPathString());
      buildInfoSize++;
    }
    for (String feature : features) {
      fp.addString(feature);
    }
    // Sharing of native dependencies may cause an {@link ActionConflictException} when ThinLTO is
    // disabled for test and test-only targets that are statically linked, but enabled for other
    // statically linked targets. This happens in case the artifacts for the shared native
    // dependency are output by {@link Action}s owned by the non-test and test targets both. To fix
    // this, we allow creation of multiple artifacts for the shared native library - one shared
    // among the test and test-only targets where ThinLTO is disabled, and the other shared among
    // other targets where ThinLTO is enabled.
    fp.addBoolean(isTestTargetInPartiallyDisabledThinLtoCase);
    return PathFragment.create(
        "_nativedeps/"
            + linkerInputsSize
            + "_"
            + linkopts.size()
            + "_"
            + linkstampsString
            + "_"
            + buildInfoSize
            + "_"
            + features.size()
            + "_"
            + fp.hexDigestAndReset());
  }
}
