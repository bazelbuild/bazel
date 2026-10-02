// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import com.google.common.collect.AbstractIterator;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterators;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import java.util.ArrayDeque;
import java.util.Iterator;

/** Input traversal for remote prefetching and concurrent modification checks. */
final class RemoteActionInputs {
  private RemoteActionInputs() {}

  /**
   * Lazily expands tree artifacts, runfiles trees and filesets into their constituent inputs.
   *
   * <p>Uses the actual runfiles mapping, excluding shadowed artifacts. Empty runfiles entries and
   * empty tree artifacts are omitted: they need no downloads, and directory metadata does not
   * detect concurrent modifications. Inputs retain their original exec paths and may occur more
   * than once. No input mapping or per-file objects are allocated.
   */
  static Iterable<ActionInput> expand(
      Iterable<? extends ActionInput> inputs, InputMetadataProvider metadataProvider) {
    return () ->
        new AbstractIterator<>() {
          // Keep iterators only for aggregates, avoiding a singleton list/iterator per leaf input.
          private final ArrayDeque<Iterator<? extends ActionInput>> iterators =
              new ArrayDeque<>(ImmutableList.of(inputs.iterator()));

          @Override
          protected ActionInput computeNext() {
            while (!iterators.isEmpty()) {
              var iterator = iterators.peek();
              if (!iterator.hasNext()) {
                iterators.pop();
                continue;
              }
              ActionInput input = iterator.next();
              switch (input) {
                case Artifact artifact when artifact.isTreeArtifact() -> {
                  var tree = metadataProvider.getTreeMetadata(artifact);
                  if (tree != null) {
                    iterators.push(tree.getChildren().iterator());
                  }
                }
                case Artifact artifact when artifact.isRunfilesTree() ->
                    // Follow the actual mapping, which excludes shadowed artifacts.
                    iterators.push(
                        metadataProvider
                            .getRunfilesMetadata(artifact)
                            .getRunfilesTree()
                            .getMapping()
                            .values()
                            .iterator());
                case Artifact artifact when artifact.isFileset() ->
                    iterators.push(
                        Iterators.transform(
                            metadataProvider.getFileset(artifact).symlinks().iterator(),
                            FilesetOutputSymlink::target));
                case null -> {
                  // Empty runfiles entries are materialized by the spawn runner.
                }
                default -> {
                  return input;
                }
              }
            }
            return endOfData();
          }
        };
  }
}
