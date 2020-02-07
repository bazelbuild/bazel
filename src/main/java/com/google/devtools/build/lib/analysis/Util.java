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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.rules.AliasConfiguredTarget;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Set;

/** Utility methods for use by ConfiguredTarget implementations. */
public abstract class Util {

  private Util() {}

  // ---------- Label and Target related methods

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
   * Returns the workspace-relative path of the specified target (file or rule), prepending a prefix
   * and appending a suffix.
   *
   * <p>For example, "//foo/bar:wiz" and "//foo:bar/wiz" both result in "foo/bar/wiz".
   */
  public static PathFragment getWorkspaceRelativePath(Target target, String prefix, String suffix) {
    return target.getLabel().getPackageFragment().getRelative(prefix + target.getName() + suffix);
  }

  /** Checks if a PathFragment contains a '-'. */
  public static boolean containsHyphen(PathFragment path) {
    return path.getPathString().indexOf('-') >= 0;
  }

  // ---------- Implicit dependency extractor

  /*
   * Given a RuleContext, find all the implicit attribute deps aka deps that weren't explicitly set
   * in the build file but are attached behind the scenes to some attribute. This means this
   * function does *not* cover deps attached other ways e.g. toolchain-related implicit deps
   * (see {@link PostAnalysisQueryEnvironment#targetifyValues} for more info on further implicit
   * deps filtering).
   * note: nodes that are depended on both implicitly and explicitly are considered explicit.
   */
  public static ImmutableSet<ConfiguredTargetKey> findImplicitDeps(RuleContext ruleContext) {
    Set<ConfiguredTargetKey> maybeImplicitDeps = CompactHashSet.create();
    Set<ConfiguredTargetKey> explicitDeps = CompactHashSet.create();
    AttributeMap attributes = ruleContext.attributes();
    ListMultimap<String, ConfiguredTargetAndData> targetMap =
        ruleContext.getConfiguredTargetAndDataMap();
    for (String attrName : attributes.getAttributeNames()) {
      List<ConfiguredTargetAndData> attrValues = targetMap.get(attrName);
      if (attrValues != null && !attrValues.isEmpty()) {
        if (attributes.isAttributeValueExplicitlySpecified(attrName)) {
          addLabelsAndConfigs(explicitDeps, attrValues);
        } else {
          addLabelsAndConfigs(maybeImplicitDeps, attrValues);
        }
      }
    }
    return ImmutableSet.copyOf(Sets.difference(maybeImplicitDeps, explicitDeps));
  }

  private static void addLabelsAndConfigs(
      Set<ConfiguredTargetKey> set, List<ConfiguredTargetAndData> deps) {
    for (ConfiguredTargetAndData dep : deps) {
      // This must be done because {@link AliasConfiguredTarget#getLabel} returns the label of the
      // "actual" configured target instead of the alias.
      if (dep.getConfiguredTarget() instanceof AliasConfiguredTarget) {
        set.add(
            ConfiguredTargetKey.of(
                ((AliasConfiguredTarget) dep.getConfiguredTarget()).getOriginalLabel(),
                dep.getConfiguration()));
      } else {
        set.add(
            ConfiguredTargetKey.of(dep.getConfiguredTarget().getLabel(), dep.getConfiguration()));
      }
    }
  }
}
