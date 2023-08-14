// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.genquery;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import javax.annotation.Nullable;

/** Factory for {@link GenQueryPackageProvider}. */
public interface GenQueryPackageProviderFactory {

  /**
   * Loads the transitive closure of {@code scope} and returns its package and target information.
   *
   * <p>Returns {@code null} if doing so requires additional Skyframe work to be done.
   *
   * @throws BrokenQueryScopeException if doing so requires looking at any packages in error.
   */
  @Nullable
  GenQueryPackageProvider constructPackageMap(Environment env, ImmutableList<Label> scope)
      throws InterruptedException, BrokenQueryScopeException;

  /**
   * Thrown if {@link GenQueryPackageProviderFactory#constructPackageMap} cannot load the full
   * transitive closure of its scope because of broken or missing packages or missing targets.
   */
  class BrokenQueryScopeException extends Exception {

    public static BrokenQueryScopeException of(@Nullable NoSuchThingException cause) {
      return cause == null ? new BrokenQueryScopeException() : new BrokenQueryScopeException(cause);
    }

    private BrokenQueryScopeException() {
      super("errors were encountered while computing transitive closure of the scope");
    }

    private BrokenQueryScopeException(NoSuchThingException cause) {
      super(
          "errors were encountered while computing transitive closure of the scope: "
              + cause.getMessage(),
          cause);
    }
  }
}
