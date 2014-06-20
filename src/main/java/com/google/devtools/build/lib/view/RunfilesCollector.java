// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.view;

// TODO(bazel-team): This class is not used and should be removed very soon.
/**
 * A helper class that collects runfiles for a given target.
 */
public abstract class RunfilesCollector {

  /**
   * States of RunfilesCollector.
   *
   * <p>These are used for the cases when the runfiles collection of a target differs depending on
   * the type of the dependency. For example, the same target might provide one set of runfiles when
   * it is depended on as a compilation dependency, and a different set when it is a data
   * dependency.
   */
  // TODO(bazel-team): delete from OSS tree
  public static enum State {
    DEFAULT,
    DATA,
    INTERPRETED,
  }
}
