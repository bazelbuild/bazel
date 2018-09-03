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
package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.events.Location;

/**
 * An instance (in the Skylark sense, not Java) of a {@link Provider}.
 *
 * <p>Info objects are specially handled in skylark, serving as units of information passed
 * between targets. Each Info object must be associated with a Provider key, defined by the
 * Provider which constructs Info objects of its type.
 */
public interface InfoInterface {

  /**
   * Returns the Skylark location where this instance was created.
   *
   * <p>Builtin provider instances may return {@link Location#BUILTIN}.
   */
  Location getCreationLoc();

  /**
   * Returns the provider instance that constructs instances of this info.
   */
  Provider getProvider();
}