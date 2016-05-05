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
package com.google.devtools.build.android;

import com.android.ide.common.res2.MergingException;

import java.io.IOException;
import java.nio.file.Path;

/**
 * An interface for visiting android data for writing.
 */
public interface AndroidDataWritingVisitor {
  /**
   * Copies the AndroidManifest to the destination directory.
   */
  Path copyManifest(Path sourceManifest) throws IOException;

  /**
   * Copies the source asset to the relative destination path.
   *
   * @param source The source file to copy.
   * @param relativeDestinationPath The relative destination path to write the asset to.
   * @throws IOException if there are errors during copying.
   */
  void copyAsset(Path source, String relativeDestinationPath) throws IOException;

  /**
   * Copies the source resource to the relative destination path.
   *
   * @param source The source file to copy.
   * @param relativeDestinationPath The relative destination path to write the resource to.
   * @throws IOException if there are errors during copying.
   * @throws MergingException for errors during png crunching.
   */
  void copyResource(Path source, String relativeDestinationPath)
      throws IOException, MergingException;

  /**
   * Adds a xml string fragment to the values file.
   *
   * @param key Used to ensure a constant order of the written xml.
   * @param xmlFragment the xml fragment as an Iterable<String> which allows lazy generation.
   */
  // TODO(corysmith): Change this to pass in a xml writer. Safer all around.
  void writeToValuesXml(FullyQualifiedName key, Iterable<String> xmlFragment);
}
