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

import javax.annotation.Nullable;

/**
 * Side-effect-free query methods for information about an action being executed. The intent is that
 * any information required about an action during its actual execution be exposed here. This should
 * mainly be functionality for printing, status messages and the like.
 */
public interface ExecutableMetadata {
  /**
   * If this executable can supply verbose information, returns a string that can be used as a
   * progress message while this executable is running. A return value of {@code null} indicates no
   * message should be reported.
   */
  @Nullable
  public String getProgressMessage();

  /**
   * Returns the owner of this executable if this executable can supply verbose information. This is
   * typically the rule that constructed it; see ActionOwner class comment for details. Returns
   * {@code null} if no owner can be determined.
   *
   * <p>If this executable does not supply verbose information, this function may throw an
   * IllegalStateException.
   */
  public ActionOwner getOwner();

  /**
   * Returns a mnemonic (string constant) for this kind of executable; written into the master log
   * so that the appropriate parser can be invoked for the output of the executable. Effectively a
   * public method as the value is used by the extra_action feature to match actions.
   */
  String getMnemonic();

  /**
   * Returns a pretty string representation of this action, suitable for use in
   * progress messages or error messages.
   */
  String prettyPrint();

  /**
   * Returns a string that can be used to describe the execution strategy. For example, "local".
   *
   * May return null if the executable chooses to update its strategy locality "manually", via
   * ActionLocalityMessage.
   *
   * @param executor the application-specific value passed to the
   *   executor parameter of the top-level call to
   *   Builder.buildArtifacts().
   */
  public String describeStrategy(Executor executor);

  /**
   * Returns the "primary" output of this executable.
   *
   * <p>For example, the linked library would be the primary output of a LinkAction.
   *
   * <p>Never returns null.
   */
  Artifact getPrimaryOutput();
}
