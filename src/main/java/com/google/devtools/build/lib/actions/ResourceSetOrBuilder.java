// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.util.OS;
import java.util.Map;

/** Common interface for ResourceSet and builder. */
@FunctionalInterface
public interface ResourceSetOrBuilder {
  /**
   * Returns resource set based on number of inputs. If build requires the size of inputs, then it
   * will flatten NestedSet. This action could create a lot of garbagge, so use it as close as
   * possible to the execution phase,
   */
  public ResourceSet buildResourceSet(OS os, int inputsSize)
      throws ExecException, InterruptedException;

  /**
   * Returns the resources a spawn with this declaration should book with the {@code
   * ResourceManager} (the built resource set), with the "resources:*" entries in {@code
   * executionInfo} and {@code execProperties} applied as overrides.
   *
   * <p>The entries are taken unparsed so that a declaration which ignores them, such as {@link
   * #fixed}, doesn't pay to parse them.
   */
  default ResourceSet buildLocalResources(
      OS os, int inputsSize, Map<String, String> executionInfo, Map<String, String> execProperties)
      throws ExecException, InterruptedException {
    return buildResourceSet(os, inputsSize)
        .withResourceOverrides(
            ExecutionRequirements.parseResources(executionInfo),
            ExecutionRequirements.parseResources(execProperties));
  }

  /**
   * Returns a declaration of {@code resources} that the owning target's "resources:*" entries
   * must not override.
   */
  static ResourceSetOrBuilder fixed(ResourceSet resources) {
    return new ResourceSetOrBuilder() {
      @Override
      public ResourceSet buildResourceSet(OS os, int inputsSize) {
        return resources;
      }

      @Override
      public ResourceSet buildLocalResources(
          OS os,
          int inputsSize,
          Map<String, String> executionInfo,
          Map<String, String> execProperties) {
        return resources;
      }
    };
  }
}
