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

import com.google.common.collect.ImmutableMap;

/** This class contains Bazel-specific functions to extend or interoperate with Skylark. */
public final class SkylarkUtils {

  /** Bazel-specific information that we store in the Environment. */
  private static class BazelInfo {
    String toolsRepository;
    ImmutableMap<String, Class<?>> fragmentNameToClass;
    Object lipoDataTransition;
  }

  private static final String BAZEL_INFO_KEY = "$bazel";

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

  /**
   * Sets, on an {@link Environment}, a {@link Map} from configuration fragment name to
   * configuration fragment class.
   */
  public static void setFragmentMap(Environment env,
      ImmutableMap<String, Class<?>> fragmentNameToClass) {
    getInfo(env).fragmentNameToClass = fragmentNameToClass;
  }

  /**
   * Returns the {@link Map} from configuration fragment name to configuration fragment class, as
   * set by {@link #setFragmentMap}.
   */
  public static ImmutableMap<String, Class<?>> getFragmentMap(Environment env) {
    return getInfo(env).fragmentNameToClass;
  }

  /**
   * Sets the configuration transition to apply to <code>cfg = "data"</code> attributes.
   *
   * <p>This must be a
   * {@link com.google.devtools.build.lib.analysis.config.transitions.PatchTransition}. But that
   * class isn't available in <code>lib.syntax</code>, so we can't type-check it here.
   */
  public static void setLipoDataTransition(Environment env, Object lipoDataTransition) {
    getInfo(env).lipoDataTransition = lipoDataTransition;
  }

  /**
   * Returns the configuration transition to apply to <code>cfg = "data"</code> attributes.
   *
   * <p>This is always a
   * {@link com.google.devtools.build.lib.analysis.config.transitions.PatchTransition}.
   */
  public static Object getLipoDataTransition(Environment env) {
    return getInfo(env).lipoDataTransition;
  }
}
