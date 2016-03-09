// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Represents an Android Resource parsed from an xml or binary file.
 */
public interface DataResource {

  /**
   * Writes the resource to the given resource directory.
   * @param newResourceDirectory The new directory for this resource.
   * @throws IOException if there are issues with writing the resource.
   */
  void write(Path newResourceDirectory) throws IOException;
}
