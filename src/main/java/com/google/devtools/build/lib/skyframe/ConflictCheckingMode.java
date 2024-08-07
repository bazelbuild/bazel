// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

/** Represents the different strategies for conflict checking in Skymeld mode. */
public enum ConflictCheckingMode {
  // No conflict check required.
  NONE,

  // Perform a traversal of the dependency graph to collect the ActionLookupValues and go through
  // the actions to look for conflicts (matches or prefix). Requires the graph to have edges.
  WITH_TRAVERSAL,

  // Before creating each ConfiguredTarget/Aspect, go through the actions to look for conflicts. A
  // conflict would then result in the failure to analyse said CT/Aspect.
  // Used when the graph has no edges.
  UPON_CONFIGURED_OBJECT_CREATION;
}
