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

import java.nio.file.Path;

/**
 * An {@link XmlResourceValue} is extracted from xml files in the resource 'values' directory.
 */
public interface XmlResourceValue {
  /**
   * Each XmlValue is expected to write a valid representation in xml to the writer.
   *
   * @param key The FullyQualified name for the xml resource being written.
   * @param source The source of the value to allow for proper comment annotation.
   * @param mergedDataWriter The target writer.
   */
  void write(FullyQualifiedName key, Path source, AndroidDataWritingVisitor mergedDataWriter);
}
