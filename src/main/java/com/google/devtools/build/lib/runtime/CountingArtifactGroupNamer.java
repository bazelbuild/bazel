// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.NamedSetOfFilesId;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.concurrent.ThreadSafe;

/** Conversion of paths to URIs. */
@ThreadSafe
public class CountingArtifactGroupNamer implements ArtifactGroupNamer {

  private final Map<NestedSet.Node, Integer> nodeNames = new HashMap<>();

  @Override
  public NamedSetOfFilesId apply(NestedSet.Node id) {
    Integer name;
    synchronized (this) {
      name = nodeNames.get(id);
    }
    if (name == null) {
      return null;
    }
    return NamedSetOfFilesId.newBuilder().setId(name.toString()).build();
  }

  /**
   * If the {@link NestedSet} has no name already, return a new name for it. Return null otherwise.
   */
  public synchronized String maybeName(NestedSet<?> set) {
    NestedSet.Node id = set.toNode();
    if (nodeNames.containsKey(id)) {
      return null;
    }
    Integer name = nodeNames.size();
    nodeNames.put(id, name);
    return name.toString();
  }
}
