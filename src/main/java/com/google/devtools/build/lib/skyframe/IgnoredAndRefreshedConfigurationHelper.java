// Copyright 2019 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;
import java.util.Collections;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Class contains logic for work with blacklist (.bazelignore) and refresh roots configurations;
 * - refreshes the configuration values from corresponding files
 * - batch-invalidates the files in directories with changed "ignored" or "refresh root" status
 *
 * Called before fine-grained computation of file differences by SequencedSkyframeExecutor, so that
 * difference checker has the up-to-date configuration.
 */
public class IgnoredAndRefreshedConfigurationHelper {
  private final ConfigurationFile blacklistConfiguration;
  private final ConfigurationFile refreshRootsConfiguration;

  IgnoredAndRefreshedConfigurationHelper(PathFragment blacklistPrefixesFile) {
    blacklistConfiguration = new ConfigurationFile(BlacklistedPackagePrefixesValue.key(),
        LabelConstants.WORKSPACE_FILE_NAME,
        value -> ((BlacklistedPackagePrefixesValue) value).getPatterns());
    refreshRootsConfiguration = new ConfigurationFile(RefreshRootsValue.key(),
        blacklistPrefixesFile,
        value -> ((RefreshRootsValue) value).getRoots().keySet());
  }

  BlacklistedPackagePrefixesValue computeBlacklist(SkyframeExecutorAdapter partner,
      Root workspaceRoot) throws InterruptedException {
    return (BlacklistedPackagePrefixesValue) new Evaluator(partner, workspaceRoot)
        .computeValue(blacklistConfiguration);
  }

  RefreshRootsValue computeRefreshRoots(SkyframeExecutorAdapter partner,
      Root workspaceRoot) throws InterruptedException {
    return (RefreshRootsValue) new Evaluator(partner, workspaceRoot)
        .computeValue(refreshRootsConfiguration);
  }

  private static class Evaluator {
    private final SkyframeExecutorAdapter partner;
    private final Root workspaceRoot;

    private Evaluator(
        SkyframeExecutorAdapter partner, Root workspaceRoot) {
      this.partner = partner;
      this.workspaceRoot = workspaceRoot;
    }

    SkyValue computeValue(ConfigurationFile file) throws InterruptedException {
      SkyValue oldValue = partner.getOld(file.key);
      refreshConfigurationFile(file.configurationFile);
      SkyValue newValue = partner.getNew(file.key);
      invalidateFragments(getRootsFromValue(file, oldValue), getRootsFromValue(file, newValue));

      return newValue;
    }

    private Collection<PathFragment> getRootsFromValue(ConfigurationFile file, SkyValue value) {
      if (value == null) {
        return Collections.emptySet();
      }
      return file.rootsProvider.apply(value);
    }

    private void refreshConfigurationFile(PathFragment configurationFile)
        throws InterruptedException {
      RootedPath path = RootedPath.toRootedPath(workspaceRoot, configurationFile);
      SkyKey fileStateKey = FileStateValue.key(path);
      SkyValue oldValue = partner.getOld(fileStateKey);
      // if we do not have the value cached, we do not need to invalidate it
      if (oldValue == null) {
        return;
      }
      boolean isDirty = partner.checkDirtiness(fileStateKey, oldValue);
      if (isDirty) {
        partner.refreshExactly(path);
      }
    }

    private void invalidateFragments(
        Collection<PathFragment> oldFragments,
        Collection<PathFragment> newFragments) throws InterruptedException {
      Set<PathFragment> onlyInOld = Sets.newHashSet(oldFragments);
      onlyInOld.removeAll(newFragments);
      Set<PathFragment> onlyInNew = Sets.newHashSet(newFragments);
      onlyInNew.removeAll(oldFragments);

      Set<RootedPath> roots = Sets.newHashSet();
      Consumer<PathFragment> adder = f -> roots.add(RootedPath.toRootedPath(workspaceRoot, f));
      onlyInOld.forEach(adder);
      onlyInNew.forEach(adder);

      if (!roots.isEmpty()) {
        partner.refreshUnder(roots);
      }
    }
  }

  private static class ConfigurationFile {
    private final SkyKey key;
    private final PathFragment configurationFile;
    private final Function<SkyValue, Set<PathFragment>> rootsProvider;

    ConfigurationFile(SkyKey key,
        PathFragment configurationFile,
        Function<SkyValue, Set<PathFragment>> rootsProvider) {
      this.key = key;
      this.configurationFile = configurationFile;
      this.rootsProvider = rootsProvider;
    }
  }

  interface SkyframeExecutorAdapter {
    SkyValue getOld(SkyKey key) throws InterruptedException;
    SkyValue getNew(SkyKey key) throws InterruptedException;
    void refreshUnder(Set<RootedPath> paths) throws InterruptedException;

    void refreshExactly(RootedPath path);

    boolean checkDirtiness(SkyKey key, SkyValue oldValue) throws InterruptedException;
  }
}
