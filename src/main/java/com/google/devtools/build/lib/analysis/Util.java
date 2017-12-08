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

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Set;

/**
 * Utility methods for use by ConfiguredTarget implementations.
 */
public abstract class Util {

  private Util() {}

  //---------- Label and Target related methods

  /**
   * Returns the workspace-relative path of the specified target (file or rule).
   *
   * <p>For example, "//foo/bar:wiz" and "//foo:bar/wiz" both result in "foo/bar/wiz".
   */
  public static PathFragment getWorkspaceRelativePath(Target target) {
    return getWorkspaceRelativePath(target.getLabel());
  }

  /**
   * Returns the workspace-relative path of the specified target (file or rule).
   *
   * <p>For example, "//foo/bar:wiz" and "//foo:bar/wiz" both result in "foo/bar/wiz".
   */
  public static PathFragment getWorkspaceRelativePath(Label label) {
    return label.getPackageFragment().getRelative(label.getName());
  }

  /**
   * Returns the workspace-relative path of the specified target (file or rule),
   * prepending a prefix and appending a suffix.
   *
   * <p>For example, "//foo/bar:wiz" and "//foo:bar/wiz" both result in "foo/bar/wiz".
   */
  public static PathFragment getWorkspaceRelativePath(Target target, String prefix, String suffix) {
    return target.getLabel().getPackageFragment().getRelative(prefix + target.getName() + suffix);
  }

  /**
   * Checks if a PathFragment contains a '-'.
   */
  public static boolean containsHyphen(PathFragment path) {
    return path.getPathString().indexOf('-') >= 0;
  }

  // ---------- Implicit dependency extractor

  /*
   * Given a RuleContext, find all the implicit deps aka deps that weren't explicitly set in the
   * build file and all toolchain deps.
   * note: nodes that are depended on both implicitly and explicitly are considered explicit.
   */
  public static ImmutableSet<LabelAndConfiguration> findImplicitDeps(RuleContext ruleContext) {
    // (1) Consider rule attribute dependencies.
    Set<LabelAndConfiguration> maybeImplicitDeps = CompactHashSet.create();
    Set<LabelAndConfiguration> explicitDeps = CompactHashSet.create();
    AttributeMap attributes = ruleContext.attributes();
    ListMultimap<String, ? extends TransitiveInfoCollection> targetMap =
        ruleContext.getConfiguredTargetMap();
    for (String attrName : attributes.getAttributeNames()) {
      List<? extends TransitiveInfoCollection> attrValues = targetMap.get(attrName);
      if (attrValues != null && !attrValues.isEmpty()) {
        if (attributes.isAttributeValueExplicitlySpecified(attrName)) {
          addLabelsAndConfigs(explicitDeps, attrValues);
        } else {
          addLabelsAndConfigs(maybeImplicitDeps, attrValues);
        }
      }
    }
    // (2) Consider toolchain dependencies
    ToolchainContext toolchainContext = ruleContext.getToolchainContext();
    if (toolchainContext != null) {
      BuildConfiguration config = ruleContext.getConfiguration();
      for (Label toolchain : toolchainContext.getResolvedToolchainLabels()) {
        maybeImplicitDeps.add(LabelAndConfiguration.of(toolchain, config));
      }
      maybeImplicitDeps.add(
          LabelAndConfiguration.of(toolchainContext.getExecutionPlatform().label(), config));
      maybeImplicitDeps.add(
          LabelAndConfiguration.of(toolchainContext.getTargetPlatform().label(), config));
    }
    return ImmutableSet.copyOf(Sets.difference(maybeImplicitDeps, explicitDeps));
  }

  private static void addLabelsAndConfigs(
      Set<LabelAndConfiguration> set, List<? extends TransitiveInfoCollection> deps) {
    deps.forEach(
        target -> set.add(LabelAndConfiguration.of(target.getLabel(), target.getConfiguration())));
  }
}
