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

import com.google.devtools.build.lib.cmdline.Label;

/** This class contains Bazel-specific functions to extend or interoperate with Skylark. */
public final class SkylarkUtils {

  /** Bazel-specific information that we store in the Environment. */
  private static class BazelInfo {
    String toolsRepository;
  }

  private static final String BAZEL_INFO_KEY = "$bazel";
  private static final String CALLER_LABEL_KEY = "$caller_label";

  private static BazelInfo getInfo(Environment env) {
    Object info = env.lookup(BAZEL_INFO_KEY);
    if (info != null) {
      return (BazelInfo) info;
    }

    BazelInfo result = new BazelInfo();
    try {
      env.update(BAZEL_INFO_KEY, result);
      return result;
    } catch (EvalException e) {
      throw new AssertionError(e);
    }
  }

  public static void setToolsRepository(Environment env, String toolsRepository) {
    getInfo(env).toolsRepository = toolsRepository;
  }

  public static String getToolsRepository(Environment env) {
    return getInfo(env).toolsRepository;
  }

  public static void setCallerLabel(Environment env, Label label) {
    // We cannot store this information in BazelInfo, because we need to
    // have it in the local environment (not the global environment).
    try {
      env.update(CALLER_LABEL_KEY, label);
    } catch (EvalException e) {
      throw new AssertionError(e);
    }
  }

  /**
   * Gets the label of the BUILD file that is using this environment. For example, if a target
   * //foo has a dependency on //bar which is a Skylark rule defined in //rules:my_rule.bzl being
   * evaluated in this environment, then this would return //foo.
   */
  public static Label getCallerLabel(Environment env) {
    return (Label) env.lookup(CALLER_LABEL_KEY);
  }
}
