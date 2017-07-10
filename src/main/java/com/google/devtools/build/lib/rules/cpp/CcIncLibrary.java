// Copyright 2016 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * An inclusion library that maps header files into a different directory. This
 * is mostly used for third party libraries that have version-specific sub-directories,
 * and also for libraries that provide architecture-dependent header files.
 * In both cases, we want normal code to be able to include files without requiring
 * the version number or the architecture name in the include statement.
 *
 * <p>Example: a <code>cc_inc_library</code> rule in
 * <code>third_party/foo</code> has the name <code>bar</code>. It will create a
 * symlink in <code>include_directory/third_party/foo/bar/third_party/foo</code>
 * pointing to <code>third_party/foo/1.0</code>. This results in the include
 * directory <code>include_directory/third_party/foo/bar</code> to be used for
 * compilations, which makes inclusions like
 * <code>#include "third_party/foo/header.h"</code> work.
 */
public abstract class CcIncLibrary implements RuleConfiguredTargetFactory {

  private final CppSemantics semantics;

  protected CcIncLibrary(CppSemantics semantics) {
    this.semantics = semantics;
  }
  
  @Override
  public ConfiguredTarget create(final RuleContext ruleContext)
      throws RuleErrorException, InterruptedException {
    CcToolchainProvider ccToolchain =
        CppHelper.getToolchainUsingDefaultCcToolchainAttribute(ruleContext);
    FeatureConfiguration featureConfiguration =
        CcCommon.configureFeatures(ruleContext, ccToolchain);
    PathFragment packageFragment = ruleContext.getPackageDirectory();

    // The rule needs a unique location for the include directory, which doesn't conflict with any
    // other rule. For that reason, the include directory is at:
    // configuration/package_name/_/target_name
    // And then the symlink is placed at:
    // configuration/package_name/_/target_name/package_name
    // So that these inclusions can be resolved correctly:
    // #include "package_name/a.h"
    //
    // The target of the symlink is:
    // package_name/targetPrefix/
    // All declared header files must be below that directory.
    String expandedIncSymlinkAttr = ruleContext.attributes().get("prefix", Type.STRING);

    // We use an additional "_" directory here to avoid conflicts between this and previous Blaze
    // versions. Previous Blaze versions created a directory symlink; the new version does not
    // detect that the output directory isn't a directory, and tries to put the symlinks into what
    // is actually a symlink into the source tree.
    PathFragment includeDirectory = PathFragment.create("_")
        .getRelative(ruleContext.getTarget().getName());
    Root configIncludeDirectory =
        ruleContext.getConfiguration().getIncludeDirectory(ruleContext.getRule().getRepository());
    PathFragment includePath =
        configIncludeDirectory
            .getExecPath()
            .getRelative(packageFragment)
            .getRelative(includeDirectory);
    Path includeRoot =
        configIncludeDirectory.getPath().getRelative(packageFragment).getRelative(includeDirectory);

    // For every source artifact, we compute a virtual artifact that is below the include directory.
    // These are used for include checking.
    PathFragment prefixFragment = packageFragment.getRelative(expandedIncSymlinkAttr);
    if (!prefixFragment.isNormalized()) {
      ruleContext.attributeWarning("prefix", "should not contain '.' or '..' elements");
    }
    ImmutableSortedMap.Builder<Artifact, Artifact> virtualArtifactMapBuilder =
        ImmutableSortedMap.orderedBy(Artifact.EXEC_PATH_COMPARATOR);
    prefixFragment = prefixFragment.normalize();
    ImmutableList<Artifact> hdrs = ruleContext.getPrerequisiteArtifacts("hdrs", Mode.TARGET).list();
    for (Artifact src : hdrs) {
      // All declared header files must start with package/targetPrefix.
      if (!src.getRootRelativePath().startsWith(prefixFragment)) {
        ruleContext.attributeError("hdrs", src + " does not start with '"
            + prefixFragment.getPathString() + "'");
        return null;
      }

      // Remove the targetPrefix from within the exec path of the source file, and prepend the
      // unique directory prefix, e.g.:
      // third_party/foo/1.2/bar/a.h -> third_party/foo/name/third_party/foo/bar/a.h
      PathFragment suffix = src.getRootRelativePath().relativeTo(prefixFragment);
      PathFragment virtualPath = includeDirectory.getRelative(packageFragment)
          .getRelative(suffix);

      // These virtual artifacts have the symlink action as generating action.
      Artifact virtualArtifact =
          ruleContext.getPackageRelativeArtifact(virtualPath, configIncludeDirectory);
      virtualArtifactMapBuilder.put(virtualArtifact, src);
    }
    ImmutableSortedMap<Artifact, Artifact> virtualArtifactMap = virtualArtifactMapBuilder.build();
    ruleContext.registerAction(
        new CreateIncSymlinkAction(ruleContext.getActionOwner(), virtualArtifactMap, includeRoot));
    FdoSupportProvider fdoSupport =
        CppHelper.getFdoSupportUsingDefaultCcToolchainAttribute(ruleContext);
    CcLibraryHelper.Info info =
        new CcLibraryHelper(ruleContext, semantics, featureConfiguration, ccToolchain, fdoSupport)
            .addIncludeDirs(Arrays.asList(includePath))
            .addPublicHeaders(virtualArtifactMap.keySet())
            .addDeps(ruleContext.getPrerequisites("deps", Mode.TARGET))
            .build();

    // cc_inc_library doesn't compile any file - no compilation outputs available.
    InstrumentedFilesProvider instrumentedFilesProvider =
          new CcCommon(ruleContext).getInstrumentedFilesProvider(
              new ArrayList<Artifact>(),
              /*withBaselineCoverage=*/true);

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProviders(info.getProviders())
        .addSkylarkTransitiveInfo(CcSkylarkApiProvider.NAME, new CcSkylarkApiProvider())
        .addOutputGroups(info.getOutputGroups())
        .add(InstrumentedFilesProvider.class, instrumentedFilesProvider)
        .add(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY))
        .build();
  }
}

