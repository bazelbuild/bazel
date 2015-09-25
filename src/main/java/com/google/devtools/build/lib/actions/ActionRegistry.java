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

import com.google.common.annotations.VisibleForTesting;

/**
 * An interface for registering actions.
 */
public interface ActionRegistry {
  /**
   * This method notifies the registry new actions.
   */
  void registerAction(Action... actions);

  /**
   * Get the (Label and BuildConfiguration) of the ConfiguredTarget ultimately responsible for all
   * these actions.
   */
  ArtifactOwner getOwner();

  /**
   * An action registry that does exactly nothing.
   */
  @VisibleForTesting
  public static final ActionRegistry NOP = new ActionRegistry() {
    @Override
    public void registerAction(Action... actions) {}

    @Override
    public ArtifactOwner getOwner() {
      return ArtifactOwner.NULL_OWNER;
    }
  };
}
