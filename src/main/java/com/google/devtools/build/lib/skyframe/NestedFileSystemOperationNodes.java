// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.Collection;
import javax.annotation.Nullable;

/** Represents multiple {@link FileSystemOperationNode}s with nestable composition. */
public final class NestedFileSystemOperationNodes implements FileSystemOperationNode {
  private final FileSystemOperationNode[] nodes;

  /**
   * Opaque storage for use by serialization.
   *
   * <p>{@link FileSystemOperationNode}, {@link FileKey} and {@link DirectoryListingKey} are
   * mutually dependent via {@link FileSystemOperationNode}. This type is opaque to avoid forcing
   * {@link FileKey} and {@link DirectoryListingKey} to depend on serialization implementation code.
   *
   * <p>The serialization implementation initializes this field with double-checked locking so it is
   * marked volatile.
   */
  private volatile Object serializationScratch;

  /**
   * Effectively, a factory method for {@link NestedFileSystemOperationNodes}, but formally a
   * factory method for {@link FileSystemOperationNode}.
   *
   * <p>When there is only one node, returns the node directly instead of creating a useless
   * wrapper.
   */
  public static FileSystemOperationNode from(Collection<FileSystemOperationNode> nodes) {
    checkArgument(!nodes.isEmpty());
    if (nodes.size() == 1) {
      return nodes.iterator().next();
    }
    return new NestedFileSystemOperationNodes(nodes.toArray(FileSystemOperationNode[]::new));
  }

  private NestedFileSystemOperationNodes(FileSystemOperationNode[] nodes) {
    this.nodes = nodes;
  }

  public int count() {
    return nodes.length;
  }

  public FileSystemOperationNode get(int index) {
    return nodes[index];
  }

  @Nullable
  public Object getSerializationScratch() {
    return serializationScratch;
  }

  public void setSerializationScratch(Object value) {
    this.serializationScratch = value;
  }
}
