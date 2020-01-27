// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.skyframe;

/**
 * What version to give an evaluated node: the max of its child versions or the graph version. Even
 * for {@link #MAX_CHILD_VERSIONS} the version may still be the graph version depending on
 * properties of the {@link SkyFunction} (if it is {@link FunctionHermeticity#NONHERMETIC}) or the
 * error state of the node.
 *
 * <p>Should be set to {@link #MAX_CHILD_VERSIONS} unless the evaluation framework is being very
 * sneaky.
 */
public enum EvaluationVersionBehavior {
  MAX_CHILD_VERSIONS,
  GRAPH_VERSION
}
