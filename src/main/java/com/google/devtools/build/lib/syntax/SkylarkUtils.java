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

/** This class contains Bazel-specific functions to extend or interoperate with Skylark. */
public final class SkylarkUtils {

  public static final String TOOLS_REPOSITORY = "$tools_repository";

  /** Unsafe version of Environment#update */
  private static void updateEnv(Environment env, String key, Object value) {
    try {
      env.update(key, value);
    } catch (EvalException e) {
      throw new AssertionError(e);
    }
  }

  public static void setToolsRepository(Environment env, String toolsRepository) {
    updateEnv(env, TOOLS_REPOSITORY, toolsRepository);
  }

  public static String getToolsRepository(Environment env) {
    return (String) env.lookup(TOOLS_REPOSITORY);
  }
}
