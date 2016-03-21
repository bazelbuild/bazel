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
package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.util.Preconditions;

import javax.annotation.Nullable;

/**
 * Contains metadata used for reporting the progress and status of an action.
 *
 * <p>Morally an action's owner is the RuleConfiguredTarget instance responsible for creating it,
 * but to avoid storing heavyweight analysis objects in actions, and to avoid coupling between the
 * analysis and actions packages, the RuleConfiguredTarget provides an instance of this class.
 */
public final class ActionOwner {
  /** An action owner for special cases. Usage is strongly discouraged. */
  public static final ActionOwner SYSTEM_ACTION_OWNER =
      new ActionOwner(null, null, "system", "empty target kind", "system", null);

  @Nullable private final Label label;
  @Nullable private final Location location;
  @Nullable private final String mnemonic;
  @Nullable private final String targetKind;
  private final String configurationChecksum;
  @Nullable private final String additionalProgressInfo;

  public ActionOwner(
      @Nullable Label label,
      @Nullable Location location,
      @Nullable String mnemonic,
      @Nullable String targetKind,
      String configurationChecksum,
      @Nullable String additionalProgressInfo) {
    this.label = label;
    this.location = location;
    this.mnemonic = mnemonic;
    this.targetKind = targetKind;
    this.configurationChecksum = Preconditions.checkNotNull(configurationChecksum);
    this.additionalProgressInfo = additionalProgressInfo;
  }

  /** Returns the location of this ActionOwner, if any; null otherwise. */
  @Nullable
  public Location getLocation() {
    return location;
  }

  /**
   * Returns the label for this ActionOwner, if any; null otherwise.
   */
  @Nullable
  public Label getLabel() {
    return label;
  }

  /** Returns the configuration's mnemonic. */
  @Nullable
  public String getConfigurationMnemonic() {
    return mnemonic;
  }

  /**
   * Returns the short cache key for the configuration of the action owner.
   *
   * <p>Special action owners that are not targets can return any string here. If the
   * underlying configuration is null, this should return "null".
   */
  public String getConfigurationChecksum() {
    return configurationChecksum;
  }

  /** Returns the target kind (rule class name) for this ActionOwner, if any; null otherwise. */
  @Nullable
  public String getTargetKind() {
    return targetKind;
  }

  /**
   * Returns additional information that should be displayed in progress messages, or {@code null}
   * if nothing should be added.
   */
  @Nullable
  String getAdditionalProgressInfo() {
    return additionalProgressInfo;
  }
}
