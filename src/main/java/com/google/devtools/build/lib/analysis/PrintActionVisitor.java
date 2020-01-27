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

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionGraphVisitor;
import com.google.devtools.build.lib.actions.ActionOwner;
import java.util.List;

/**
 * A bipartite graph visitor which accumulates actions with matching mnemonics for a target.
 */
public final class PrintActionVisitor extends ActionGraphVisitor {
  private final ConfiguredTarget target;
  private final List<ActionAnalysisMetadata> actions;
  private final Predicate<ActionAnalysisMetadata> actionMnemonicMatcher;
  private final String targetConfigurationChecksum;

  /**
   * Creates a new visitor for the actions associated with the given target that have a matching
   * mnemonic.
   */
  public PrintActionVisitor(ActionGraph actionGraph, ConfiguredTarget target,
      Predicate<ActionAnalysisMetadata> actionMnemonicMatcher) {
    super(actionGraph);
    this.target = target;
    this.actionMnemonicMatcher = actionMnemonicMatcher;
    actions = Lists.newArrayList();
    targetConfigurationChecksum = target.getConfigurationChecksum();
  }

  @Override
  protected boolean shouldVisit(ActionAnalysisMetadata action) {
    ActionOwner owner = action.getOwner();
    return owner != null && target.getLabel().equals(owner.getLabel())
        && targetConfigurationChecksum.equals(owner.getConfigurationChecksum());
  }

  @Override
  protected void visitAction(ActionAnalysisMetadata action) {
    if (actionMnemonicMatcher.apply(action)) {
      actions.add(action);
    }
  }

  /** Retrieves the collected actions since this method was last called and clears the list. */
  public ImmutableList<ActionAnalysisMetadata> getActions() {
    return ImmutableList.copyOf(actions);
  }
}
