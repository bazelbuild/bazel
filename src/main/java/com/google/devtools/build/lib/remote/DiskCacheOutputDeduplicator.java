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

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileContentsProxy;
import com.google.devtools.build.lib.actions.cache.OutputMetadataStore;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.remote.disk.DiskCacheClient;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicLong;

/** Reuses cached file contents after output validation, before action metadata is published. */
final class DiskCacheOutputDeduplicator {
  private final DiskCacheClient cache;
  private final EventHandler reporter;
  private final AtomicLong files = new AtomicLong();
  private final AtomicLong bytes = new AtomicLong();

  DiskCacheOutputDeduplicator(DiskCacheClient cache, EventHandler reporter) {
    this.cache = cache;
    this.reporter = reporter;
  }

  void finalizeAction(Action action, OutputMetadataStore metadataStore)
      throws IOException, InterruptedException {
    for (Artifact output : action.getOutputs()) {
      if (metadataStore.artifactOmitted(output) || output.isSymlink()) {
        continue;
      }
      if (output.isTreeArtifact()) {
        var parent = (SpecialArtifact) output;
        var tree = metadataStore.getTreeArtifactValue(parent);
        // A resolved tree may refer to files owned by another action.
        if (tree.getResolvedPath().isPresent()) {
          continue;
        }
        var builder = TreeArtifactValue.newBuilder(parent);
        boolean changed = false;
        for (var entry : tree.getChildValues().entrySet()) {
          var metadata = deduplicate(entry.getKey().getPath(), entry.getValue());
          changed |= metadata != entry.getValue();
          builder.putChild(entry.getKey(), metadata);
        }
        var archived = tree.getArchivedRepresentation();
        if (archived.isPresent()) {
          var archive = archived.get();
          var metadata =
              deduplicate(
                  archive.archivedTreeFileArtifact().getPath(), archive.archivedFileValue());
          changed |= metadata != archive.archivedFileValue();
          builder.setArchivedRepresentation(archive.archivedTreeFileArtifact(), metadata);
        }
        if (changed) {
          metadataStore.injectTree(parent, builder.build());
        }
      } else {
        var previous = metadataStore.getOutputMetadata(output);
        if (previous != null) {
          var metadata = deduplicate(output.getPath(), previous);
          if (metadata != previous) {
            metadataStore.injectFile(output, metadata);
          }
        }
      }
    }
  }

  FileArtifactValue deduplicate(Path output, FileArtifactValue metadata) throws IOException {
    if (!metadata.getType().isFile()
        || metadata.isRemote()
        || metadata.getContentsProxy() == null
        || metadata.getResolvedPath() != null
        || metadata.getDigest() == null
        || metadata.getSize() == 0) {
      return metadata;
    }
    // getOutputMetadata has already hashed and normalized permissions on this file. Reuse its
    // digest, and replace the contents proxy after the rename instead of hashing the clone again.
    if (!metadata.getContentsProxy().equals(FileContentsProxy.create(output.stat()))) {
      // Do not overwrite a file that changed after validation, even if its size is unchanged.
      return metadata;
    }
    FileStatus stat =
        cache.cloneOutputIfPresent(
            DigestUtil.buildDigest(metadata.getDigest(), metadata.getSize()), output);
    if (stat == null) {
      return metadata;
    }
    files.incrementAndGet();
    bytes.addAndGet(metadata.getSize());
    return FileArtifactValue.createForNormalFile(
        metadata.getDigest(), FileContentsProxy.create(stat), metadata.getSize());
  }

  void report() {
    reporter.handle(
        Event.info(
            String.format(
                "Disk cache output deduplication: replaced %d files (%d logical bytes).",
                files.get(), bytes.get())));
  }
}
