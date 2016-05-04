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

import com.android.ide.common.res2.MergingException;

import java.io.IOException;

/**
 * Represents an Android Resource parsed from an xml or binary file.
 */
public interface DataResource extends DataValue {
  /**
   * Write as a resource using the supplied {@link MergeDataWriter}.
   */
  void writeResource(FullyQualifiedName key, AndroidDataWritingVisitor mergedDataWriter)
      throws IOException, MergingException;
}
