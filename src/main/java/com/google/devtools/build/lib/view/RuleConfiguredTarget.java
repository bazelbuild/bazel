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

package com.google.devtools.build.lib.view;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.syntax.FilesetEntry;
import com.google.devtools.build.lib.syntax.Label;

import java.util.List;

import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;

/**
 * An abstract implementation of ConfiguredTarget for Rules.
 */
// TODO(bazel-team): Remove ActionOwner from here; use ruleContext.getActionOwner() instead.
public abstract class RuleConfiguredTarget extends AbstractConfiguredTarget
    implements ActionOwner {

  /**
   * The configuration transition for an attribute through which a prerequisite
   * is requested.
   */
  public enum Mode {
    TARGET,
    HOST,
    DATA,
    DONT_CHECK
  }

  protected RuleConfiguredTarget(RuleContext ruleContext) {
    super(ruleContext);
    if (ruleContext.getRule().containsErrors()) {
      throw new IllegalStateException("It is unsound to attempt to construct "
          + "a view from rules that contain errors (e.g., " + ruleContext.getRule().getLabel()
          + ").  Applications should not ignore a return code of false from "
          + "BuildView.visitTransitiveClosure.");
    }
  }

  @Override
  public Rule getTarget() {
    return (Rule) super.getTarget();
  }

  public final Rule getRule() {
    return getTarget();
  }

  @Override
  public Label getLabel() {
    return getTarget().getLabel();
  }

  @Override
  public final String getTargetKind() {
    return getTarget().getTargetKind();
  }

  /**
   * There may be operations that need to be executed on the outputs of the target, like deferred
   * dependency checking. The actions that perform these operations need to be executed
   * independently of the dependency tree, and usually only have a stamp file output to indicate the
   * end of execution. These stamp files should be added to the list of desired toplevel artifacts.
   *
   * @return the list of stamp files that should always be built
   */
  @Nullable
  public ImmutableList<Artifact> getMandatoryStampFiles() {
    return null;
  }

  @Override
  public final String getConfigurationName() {
    return getConfiguration().getShortName();
  }

  @Override
  public final String getConfigurationMnemonic() {
    return getConfiguration().getMnemonic();
  }

  @Override
  public final String getConfigurationShortCacheKey() {
    return getConfiguration().shortCacheKey();
  }

  @Override
  public final Location getLocation() {
    return getTarget().getLocation();
  }

  @Override
  public final String getAdditionalProgressInfo() {
    return getConfiguration().isHostConfiguration() ? "for host" : null;
  }

  /**
   * The configured version of FilesetEntry.
   */
  @Immutable
  public static final class ConfiguredFilesetEntry {
    private final FilesetEntry entry;
    private final TransitiveInfoCollection src;
    private final ImmutableList<TransitiveInfoCollection> files;

    ConfiguredFilesetEntry(FilesetEntry entry, TransitiveInfoCollection src) {
      this.entry = entry;
      this.src = src;
      this.files = null;
    }

    ConfiguredFilesetEntry(FilesetEntry entry, ImmutableList<TransitiveInfoCollection> files) {
      this.entry = entry;
      this.src = null;
      this.files = files;
    }

    public FilesetEntry getEntry() {
      return entry;
    }

    public TransitiveInfoCollection getSrc() {
      return src;
    }

    /**
     * Targets from FilesetEntry.files, or null if the user omitted it.
     */
    @Nullable
    public List<TransitiveInfoCollection> getFiles() {
      return files;
    }
  }
}
