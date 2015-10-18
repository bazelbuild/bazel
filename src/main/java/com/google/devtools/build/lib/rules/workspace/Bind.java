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

package com.google.devtools.build.lib.rules.workspace;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.UnmodifiableIterator;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.SkylarkProviders;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;

/**
 * Implementation for the bind rule.
 */
public class Bind implements RuleConfiguredTargetFactory {

  /**
   * This configured target pretends to be whatever type of target "actual" is, returning its
   * transitive info providers and target, but returning the label for the //external target.
   */
  private static class BindConfiguredTarget implements ConfiguredTarget, ClassObject {

    private Label label;
    private ConfiguredTarget configuredTarget;
    private BuildConfiguration config;

    BindConfiguredTarget(RuleContext ruleContext) {
      label = ruleContext.getRule().getLabel();
      config = ruleContext.getConfiguration();
      // TODO(bazel-team): we should special case ConfiguredTargetFactory.createConfiguredTarget,
      // not cast down here.
      configuredTarget = (ConfiguredTarget) ruleContext.getPrerequisite("actual", Mode.TARGET);
    }

    @Override
    public <P extends TransitiveInfoProvider> P getProvider(Class<P> provider) {
      return configuredTarget == null ? null : configuredTarget.getProvider(provider);
    }

    @Override
    public Label getLabel() {
      return label;
    }

    @Override
    public Object get(String providerKey) {
      return configuredTarget == null ? null : configuredTarget.get(providerKey);
    }

    @Override
    public UnmodifiableIterator<TransitiveInfoProvider> iterator() {
      return configuredTarget == null
          ? ImmutableList.<TransitiveInfoProvider>of().iterator()
          : configuredTarget.iterator();
    }

    @Override
    public Target getTarget() {
      return configuredTarget == null ? null : configuredTarget.getTarget();
    }

    @Override
    public BuildConfiguration getConfiguration() {
      return config;
    }

    /* ClassObject methods */

    @Override
    public Object getValue(String name) {
      if (name.equals("label")) {
        return getLabel();
      } else if (name.equals("files")) {
        // A shortcut for files to build in Skylark. FileConfiguredTarget and RunleConfiguredTarget
        // always has FileProvider and Error- and PackageGroupConfiguredTarget-s shouldn't be
        // accessible in Skylark.
        return SkylarkNestedSet.of(Artifact.class, configuredTarget == null
            ? NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER)
            : getProvider(FileProvider.class).getFilesToBuild());
      }
      return configuredTarget == null ? null : configuredTarget.get(name);
    }

    @SuppressWarnings("cast")
    @Override
    public ImmutableCollection<String> getKeys() {
      ImmutableList.Builder<String> result = ImmutableList.<String>builder().add("label", "files");
      if (configuredTarget != null) {
          result.addAll(
              configuredTarget.getProvider(SkylarkProviders.class).getKeys());
      }
      return result.build();
    }

    @Override
    public String errorMessage(String name) {
      // Use the default error message.
      return null;
    }
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    if (ruleContext.getPrerequisite("actual", Mode.TARGET) == null) {
      ruleContext.ruleError(String.format("The external label '%s' is not bound to anything",
          ruleContext.getLabel()));
      return null;
    }
    return new BindConfiguredTarget(ruleContext);
  }
}
