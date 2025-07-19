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
}
