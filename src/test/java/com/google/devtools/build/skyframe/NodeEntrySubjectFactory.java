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
package com.google.devtools.build.skyframe;

import com.google.common.truth.FailureStrategy;
import com.google.common.truth.SubjectFactory;
import com.google.common.truth.Truth;

/** {@link SubjectFactory} for {@link NodeEntry} objects, providing {@link NodeEntrySubject}s. */
public class NodeEntrySubjectFactory extends SubjectFactory<NodeEntrySubject, NodeEntry> {
  public static NodeEntrySubject assertThatNodeEntry(NodeEntry nodeEntry) {
    return Truth.assertAbout(new NodeEntrySubjectFactory()).that(nodeEntry);
  }

  @Override
  public NodeEntrySubject getSubject(FailureStrategy failureStrategy, NodeEntry nodeEntry) {
    return new NodeEntrySubject(failureStrategy, nodeEntry);
  }
}
