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
package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.Label;

/**
 * The owner of an action is responsible for reporting conflicts in the action
 * graph (two actions attempting to generate the same artifact).
 *
 * Typically an action's owner is the RuleConfiguredTarget instance responsible
 * for creating it, but to avoid coupling between the view and actions
 * packages, the RuleConfiguredTarget is hidden behind this interface, which
 * exposes only the error reporting functionality.
 */
public interface ActionOwner {

  /**
   * Returns the location of this ActionOwner, if any; null otherwise.
   */
  Location getLocation();

  /**
   * Returns the label for this ActionOwner, if any; null otherwise.
   */
  Label getLabel();

  /**
   * Returns the name of the configuration of the action owner.
   */
  String getConfigurationName();

  /**
   * Returns the configuration's mnemonic.
   */
  String getConfigurationMnemonic();

  /**
   * Returns the short cache key for the configuration of the action owner.
   *
   * <p>Special action owners that are not targets can return any string here as long as it is
   * constant. If the configuration is null, this should return "null".
   *
   * <p>These requirements exist so that {@link ActionOwner} instances are consistent with
   * {@code BuildView.ActionOwnerIdentity(ConfiguredTargetValue)}.
   */
  String getConfigurationChecksum();

  /**
   * Returns the target kind (rule class name) for this ActionOwner, if any; null otherwise.
   */
  String getTargetKind();

  /**
   * Returns additional information that should be displayed in progress messages, or {@code null}
   * if nothing should be added.
   */
  String getAdditionalProgressInfo();
}
