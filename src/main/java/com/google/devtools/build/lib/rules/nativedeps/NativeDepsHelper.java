// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Util;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.cpp.CcLinkParams;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppBuildInfo;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppHelper;
import com.google.devtools.build.lib.rules.cpp.CppLinkAction;
import com.google.devtools.build.lib.rules.cpp.Link.LinkStaticness;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * Helper class to create a dynamic library for rules which support integration with native code.
 *
 * <p>This library gets created by the build system by linking all C++ libraries in the transitive
 * closure of the dependencies into a standalone dynamic library, with some exceptions. It usually
 * does not include neverlink libraries or C++ binaries (or their transitive dependencies). Note
 * that some rules are implicitly neverlink.
 */
public abstract class NativeDepsHelper {

  private NativeDepsHelper() {}

  /**
   * Creates an Action to create a dynamic library by linking all native code
   * (C/C++) libraries in the transitive dependency closure of a rule.
   *
   * <p>This is used for all native code in a rule's dependencies, regardless of type.
   *
   * <p>We link the native deps in the equivalent of linkstatic=1, linkshared=1
   * mode.
   *
   * <p>linkstatic=1 means mostly-static mode, i.e. we select the ".a" (or
   * ".pic.a") files, but we don't include "-static" in linkopts.
   *
   * <p>linkshared=1 means we prefer the ".pic.a" files to the ".a" files, and
   * the LinkTargetType is set to DYNAMIC_LIBRARY which causes Link.java to
   * include "-shared" in the linker options.
   *
   * @param ruleContext the rule context to determine the native library
   * @param linkParams the {@link CcLinkParams} for the rule, collected with
   *        linkstatic = 1 and linkshared = 1
   * @param extraLinkOpts additional parameters to the linker
   * @param includeMalloc whether or not to link in the rule's malloc dependency
   * @param configuration the {@link BuildConfiguration} to run the link action under
   * @return the native library runfiles. If the transitive deps closure of
   *         the rule contains no native code libraries, its fields are null.
   */
  public static NativeDepsRunfiles maybeCreateNativeDepsAction(final RuleContext ruleContext,
      CcLinkParams linkParams, Collection<String> extraLinkOpts, boolean includeMalloc,
      BuildConfiguration configuration) {
    PathFragment relativePath = Util.getWorkspaceRelativePath(ruleContext.getRule());
    PathFragment nativeDepsPath = relativePath.replaceName(
        relativePath.getBaseName() + Constants.NATIVE_DEPS_LIB_SUFFIX + ".so");
    if (includeMalloc) {
      // Add in the custom malloc dependency if it was requested.
      CcLinkParams.Builder linkParamsBuilder = CcLinkParams.builder(true, true);
      linkParamsBuilder.addTransitiveArgs(linkParams);
      linkParamsBuilder.addTransitiveTarget(CppHelper.mallocForTarget(ruleContext));
      linkParams = linkParamsBuilder.build();
    }
    return maybeCreateNativeDepsAction(ruleContext, linkParams, extraLinkOpts, configuration,
        CppHelper.getToolchain(ruleContext), nativeDepsPath,
        ruleContext.getConfiguration().getBinDirectory());
  }

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
   * @param ruleContext the rule context to determine the native deps library
   * @param linkParams the {@link CcLinkParams} for the rule, collected with linkstatic = 1 and
   *        linkshared = 1
   * @return the native deps library runfiles. If the transitive deps closure of the rule contains
   *         no native code libraries, its fields are null.
   */
  public static Artifact maybeCreateAndroidNativeDepsAction(final RuleContext ruleContext,
      CcLinkParams linkParams, BuildConfiguration configuration, CcToolchainProvider toolchain) {
    PathFragment uniquePath = ruleContext.getUniqueDirectory(ANDROID_UNIQUE_DIR);
    PathFragment nativeDepsPath =
        uniquePath.replaceName("lib" + uniquePath.getBaseName() + ".so");
    return maybeCreateNativeDepsAction(
        ruleContext, linkParams, /** extraLinkOpts */ ImmutableList.<String>of(),
        configuration, toolchain, nativeDepsPath, configuration.getBinDirectory()).getLibrary();
  }

  private static NativeDepsRunfiles maybeCreateNativeDepsAction(final RuleContext ruleContext,
      CcLinkParams linkParams, Collection<String> extraLinkOpts, BuildConfiguration configuration,
      CcToolchainProvider toolchain, PathFragment nativeDepsPath, Root bindirIfShared) {
    if (linkParams.getLibraries().isEmpty()) {
      return NativeDepsRunfiles.EMPTY;
    }

    List<String> linkopts = new ArrayList<>(extraLinkOpts);
    linkopts.addAll(linkParams.flattenedLinkopts());

    Map<Artifact, ImmutableList<Artifact>> linkstamps =
        CppHelper.resolveLinkstamps(ruleContext, linkParams);
    List<Artifact> buildInfoArtifacts = linkstamps.isEmpty()
        ? ImmutableList.<Artifact>of()
        : ruleContext.getBuildInfo(CppBuildInfo.KEY);

    boolean shareNativeDeps = configuration.getFragment(CppConfiguration.class).shareNativeDeps();
    NestedSet<LibraryToLink> linkerInputs = linkParams.getLibraries();
    PathFragment linkerOutputPath = shareNativeDeps
        ? getSharedNativeDepsPath(LinkerInputs.toLibraryArtifacts(linkerInputs),
            linkopts, linkstamps.keySet(), buildInfoArtifacts,
            ruleContext.getFeatures())
        : nativeDepsPath;

    CppLinkAction.Builder builder = new CppLinkAction.Builder(
        ruleContext, linkerOutputPath, configuration, toolchain);
    CppLinkAction linkAction = builder
        .setCrosstoolInputs(toolchain.getLink())
        .addLibraries(linkerInputs)
        .setLinkType(LinkTargetType.DYNAMIC_LIBRARY)
        .setLinkStaticness(LinkStaticness.MOSTLY_STATIC)
        .addLinkopts(linkopts)
        .setNativeDeps(true)
        .setRuntimeInputs(
            toolchain.getDynamicRuntimeLinkMiddleman(), toolchain.getDynamicRuntimeLinkInputs())
        .addLinkstamps(linkstamps)
        .build();

    ruleContext.registerAction(linkAction);
    final Artifact linkerOutput = linkAction.getPrimaryOutput();

    List<Artifact> runtimeSymlinks = new LinkedList<>();

    if (shareNativeDeps) {
      // Collect dynamic-linker-resolvable symlinks for C++ runtime library dependencies.
      // Note we only need these symlinks when --share_native_deps is on, as shared native deps
      // mangle path names such that the library's conventional _solib RPATH entry
      // no longer resolves (because the target directory's relative depth gets lost).
      for (final Artifact runtimeInput : toolchain.getDynamicRuntimeLinkInputs()) {
        final Artifact runtimeSymlink = ruleContext.getAnalysisEnvironment().getDerivedArtifact(
            getRuntimeLibraryPath(ruleContext, runtimeInput), bindirIfShared);
        // Since runtime library symlinks are underneath the target's output directory and
        // multiple targets may share the same output directory, we need to make sure this
        // symlink's generating action is only set once.
        ruleContext.registerAction(
            new SymlinkAction(ruleContext.getActionOwner(), runtimeInput, runtimeSymlink, null));
        runtimeSymlinks.add(runtimeSymlink);
      }

      Artifact symlink = ruleContext.getAnalysisEnvironment().getDerivedArtifact(
          nativeDepsPath, bindirIfShared);
      ruleContext.registerAction(
          new SymlinkAction(ruleContext.getActionOwner(), linkerOutput, symlink, null));
      return new NativeDepsRunfiles(symlink, runtimeSymlinks);
    }

    return new NativeDepsRunfiles(linkerOutput, runtimeSymlinks);
  }

  /**
   * Returns the path, relative to the runfiles prefix, of the native executable
   * for the specified rule, i.e. "<package>/<rule><NATIVE_DEPS_LIB_SUFFIX>"
   */
  public static PathFragment getExecutablePath(RuleContext ruleContext) {
    PathFragment relativePath = Util.getWorkspaceRelativePath(ruleContext.getRule());
    return relativePath.replaceName(relativePath.getBaseName() + Constants.NATIVE_DEPS_LIB_SUFFIX);
  }

  /**
   * Returns the path, relative to the runfiles prefix, of a runtime library
   * symlink for the native library for the specified rule.
   */
  private static PathFragment getRuntimeLibraryPath(RuleContext ruleContext, Artifact lib) {
    PathFragment relativePath = Util.getWorkspaceRelativePath(ruleContext.getRule());
    PathFragment libParentDir =
        relativePath.replaceName(lib.getExecPath().getParentDirectory().getBaseName());
    String libName = lib.getExecPath().getBaseName();
    return new PathFragment(libParentDir, new PathFragment(libName));
  }

  /**
   * Returns the path of the shared native library. The name must be
   * generated based on the rule-specific inputs to the link actions. At this
   * point this includes order-sensitive list of linker inputs and options
   * collected from the transitive closure and linkstamp-related artifacts that
   * are compiled during linking. All those inputs can be affected by modifying
   * target attributes (srcs/deps/stamp/etc). However, target build
   * configuration can be ignored since it will either change output directory
   * (in case of different configuration instances) or will not affect anything
   * (if two targets use same configuration). Final goal is for all native
   * libraries that use identical linker command to use same output name.
   *
   * <p>TODO(bazel-team): (2010) Currently process of identifying parameters that can
   * affect native library name is manual and should be kept in sync with the
   * code in the CppLinkAction.Builder/CppLinkAction/Link classes which are
   * responsible for generating linker command line. Ideally we should reuse
   * generated command line for both purposes - selecting a name of the
   * native library and using it as link action payload. For now, correctness
   * of the method below is only ensured by validations in the
   * CppLinkAction.Builder.build() method.
   */
  private static PathFragment getSharedNativeDepsPath(Iterable<Artifact> linkerInputs,
      Collection<String> linkopts, Iterable<Artifact> linkstamps,
      Iterable<Artifact> buildInfoArtifacts, Collection<String> features) {
    Fingerprint fp = new Fingerprint();
    for (Artifact input : linkerInputs) {
      fp.addString(input.getExecPathString());
    }
    fp.addStrings(linkopts);
    for (Artifact input : linkstamps) {
      fp.addString(input.getExecPathString());
    }
    for (Artifact input : buildInfoArtifacts) {
      fp.addString(input.getExecPathString());
    }
    for (String feature : features) {
      fp.addStrings(feature);
    }
    return new PathFragment(
        Constants.NATIVE_DEPS_LIB_SUFFIX + "/" + fp.hexDigestAndReset() + ".so");
  }
}
