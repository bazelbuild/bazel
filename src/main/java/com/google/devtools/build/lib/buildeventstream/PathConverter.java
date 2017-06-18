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
package com.google.devtools.build.lib.buildeventstream;

import com.google.devtools.build.lib.vfs.Path;

/**
 * Interface for conversion of paths to URIs.
 */
public interface PathConverter {
  /**
   * Return the URI corresponding to the given path, if the path can be converted to a URI by this
   * path converter; return {@link null} otherwise.
   */
  String apply(Path path);
}
