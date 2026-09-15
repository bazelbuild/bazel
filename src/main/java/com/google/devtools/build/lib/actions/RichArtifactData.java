// Copyright 2025 The Bazel Authors. All rights reserved.
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

/**
 * Artifact contents more than just a blob of bytes.
 *
 * <p>Used when one needs to propagate structured information upwards in the dependency graph from
 * an action to those that depend on it. For example, the structure of runfiles trees is represented
 * this way: an action represents the creation of the runfiles tree and its output is the runfiles
 * tree artifact, which has rich artifact data attached which in turn contains the mapping from path
 * in the runfiles tree to the artifact that lives there.
 */
public interface RichArtifactData {}
