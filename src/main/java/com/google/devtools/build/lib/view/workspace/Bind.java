// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.view.workspace;

import com.google.common.collect.UnmodifiableIterator;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.TransitiveInfoProvider;
import com.google.devtools.build.lib.view.config.BuildConfiguration;

/**
 * Implementation for the bind rule.
 */
public class Bind implements RuleConfiguredTargetFactory {

  /**
   * This configured target pretends to be whatever type of target "actual" is, returning its
   * transitive info providers and target, but returning the label for the //external target.
   */
  private static class BindConfiguredTarget implements ConfiguredTarget {

    private Label label;
    private ConfiguredTarget configuredTarget;
    private BuildConfiguration config;

    BindConfiguredTarget(RuleContext ruleContext) {
      label = ruleContext.getRule().getLabel();
      config = ruleContext.getConfiguration();
      // TODO(bazel-team): we should special case ConfiguredTargetFactory.createAndInitialize, not
      // cast down here.
      configuredTarget = (ConfiguredTarget) ruleContext.getPrerequisite("actual", Mode.TARGET);
    }

    @Override
    public <P extends TransitiveInfoProvider> P getProvider(Class<P> provider) {
      return configuredTarget.getProvider(provider);
    }

    @Override
    public Label getLabel() {
      return label;
    }

    @Override
    public Object get(String providerKey) {
      return configuredTarget.get(providerKey);
    }

    @Override
    public UnmodifiableIterator<TransitiveInfoProvider> iterator() {
      return configuredTarget.iterator();
    }

    @Override
    public Target getTarget() {
      return configuredTarget.getTarget();
    }

    @Override
    public BuildConfiguration getConfiguration() {
      return config;
    }
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    return new BindConfiguredTarget(ruleContext);
  }

}
