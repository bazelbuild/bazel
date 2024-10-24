// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.util.Fingerprint;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/**
 * Partial implementation of {@link ActionAnalysisMetadata} to ensure consistent {@linkplain #getKey
 * action key} computation.
 */
public abstract class ActionKeyComputer implements ActionAnalysisMetadata {

  /**
   * Integer embedded in every action key.
   *
   * <p>The purpose of this member and associated property is to allow to easily invalidate the
   * action cache in case we want to mitigate bugs resulting with false-sharing.
   */
  private static final int ACTION_KEY_UNIQUIFIER =
      Integer.parseInt(System.getProperty("ACTION_KEY_UNIQUIFIER", "0"));

  @Override
  public final String getKey(
      ActionKeyContext actionKeyContext, @Nullable ArtifactExpander artifactExpander)
      throws InterruptedException {
    Fingerprint fp = new Fingerprint();

    try {
      computeKey(actionKeyContext, artifactExpander, fp);
    } catch (CommandLineExpansionException | EvalException e) {
      return KEY_ERROR;
    }

    PlatformInfo executionPlatform = getExecutionPlatform();
    if (executionPlatform == null) {
      fp.addBoolean(false);
    } else {
      fp.addBoolean(true);
      executionPlatform.addTo(fp);
    }

    return fp.addStringMap(getExecProperties()).addInt(ACTION_KEY_UNIQUIFIER).hexDigestAndReset();
  }

  /**
   * See the javadoc for {@link Action} and {@link ActionAnalysisMetadata#getKey} for the contract
   * of this method.
   */
  protected abstract void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable ArtifactExpander artifactExpander,
      Fingerprint fp)
      throws CommandLineExpansionException, EvalException, InterruptedException;
}
