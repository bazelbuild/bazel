// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.syntax;

import com.google.devtools.build.lib.events.Location;

/** This class contains Bazel-specific functions to extend or interoperate with Skylark. */
public final class SkylarkUtils {

  private static final String PHASE_KEY = "$phase";

  /** A phase for enabling or disabling certain builtin functions */
  public enum Phase {
    WORKSPACE,
    LOADING,
    ANALYSIS
  }

  public static void setPhase(Environment env, Phase phase) {
    env.setupDynamic(PHASE_KEY, phase);
  }

  private static Phase getPhase(Environment env) {
    Phase phase = (Phase) env.dynamicLookup(PHASE_KEY);
    return phase == null ? Phase.ANALYSIS : phase;
  }

  /**
   * Checks that the current Environment is in the loading or the workspace phase.
   *
   * @param symbol name of the function being only authorized thus.
   */
  public static void checkLoadingOrWorkspacePhase(Environment env, String symbol, Location loc)
      throws EvalException {
    if (getPhase(env) == Phase.ANALYSIS) {
      throw new EvalException(loc, symbol + "() cannot be called during the analysis phase");
    }
  }

  /**
   * Checks that the current Environment is in the loading phase.
   *
   * @param symbol name of the function being only authorized thus.
   */
  public static void checkLoadingPhase(Environment env, String symbol, Location loc)
      throws EvalException {
    if (getPhase(env) != Phase.LOADING) {
      throw new EvalException(loc, symbol + "() can only be called during the loading phase");
    }
  }
}
