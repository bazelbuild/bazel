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
import com.google.devtools.build.lib.cmdline.RepositoryName;
import java.util.HashMap;
import java.util.Objects;

/** This class contains Bazel-specific functions to extend or interoperate with Skylark. */
public final class SkylarkUtils {

  /** Bazel-specific information that we store in the Environment. */
  private static class BazelInfo {
    String toolsRepository;
    ImmutableMap<String, Class<?>> fragmentNameToClass;
    HashMap<RepositoryName, RepositoryName> repositoryMapping;

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof BazelInfo)) {
        return false;
      }
      BazelInfo that = (BazelInfo) obj;
      return Objects.equals(this.toolsRepository, that.toolsRepository)
          && Objects.equals(this.fragmentNameToClass, that.fragmentNameToClass)
          && Objects.equals(this.repositoryMapping, that.repositoryMapping);
    }

    @Override
    public int hashCode() {
      return Objects.hash(toolsRepository, fragmentNameToClass, repositoryMapping);
    }
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

  /** Sets the map of repository names to be changed as specified in the main WORKSPACE file. */
  public static void setRepositoryMapping(Environment env, HashMap<RepositoryName, RepositoryName> repositoryMapping) {
    getInfo(env).repositoryMapping = repositoryMapping;
  }

  /** Returns the map of repository names to be changed as specified in the main WORKSPACE file. */
  public static HashMap<RepositoryName, RepositoryName> getRepositoryMapping(Environment env) {
    return getInfo(env).repositoryMapping;
  }
}
