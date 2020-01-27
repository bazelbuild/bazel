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

package com.google.devtools.build.lib.skylarkbuildapi.core;

import com.google.common.collect.ImmutableMap;

/**
 * A helper for registering a portion of the build API to skylark environment globals.
 *
 * <p>A global environment may be initialized by tabulating globals into a single map by passing
 * a single map builder to {@link #addBindingsToBuilder} for several bootstrap helpers.
 */
public interface Bootstrap {

  /** Adds this bootstrap's bindings to the given environment map builder. */
  void addBindingsToBuilder(ImmutableMap.Builder<String, Object> builder);
}
