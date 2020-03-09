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

import com.android.aapt.Resources.Reference;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.AndroidResourceMerger.MergingException;
import com.google.devtools.build.android.resources.Visibility;

/** Represents an Android Resource parsed from an xml or binary file. */
public interface DataResource extends DataValue {
  /** Write as a resource using the supplied {@link AndroidDataWritingVisitor}. */
  void writeResource(FullyQualifiedName key, AndroidDataWritingVisitor writer)
      throws MergingException;

  /**
   * Combines these resource together and returns a single resource.
   *
   * @param resource Another resource to be combined with this one.
   * @return A union of the values of these two resources.
   * @throws IllegalArgumentException if either resource cannot combine with the other.
   */
  DataResource combineWith(DataResource resource);

  /** Queue up writing the resource to the given {@link AndroidResourceSymbolSink}. */
  void writeResourceToClass(FullyQualifiedName key, AndroidResourceSymbolSink sink);

  /** Overwrite another {@link DataResource}. */
  DataResource overwrite(DataResource other);

  /** Visibility of this resource as denoted by a {@code <public>} tag, or lack thereof. */
  Visibility getVisibility();

  /** Resources referenced via XML attributes or proxying resource definitions. */
  ImmutableList<Reference> getReferencedResources();
}
