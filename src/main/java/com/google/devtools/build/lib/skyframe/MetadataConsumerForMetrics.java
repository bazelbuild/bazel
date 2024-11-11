// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileStateType;
import com.google.devtools.build.lib.actions.FilesetOutputTree;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics.ArtifactMetrics;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/** Sink for file-related metadata to be used for metrics gathering. */
@ThreadSafety.ThreadSafe
public interface MetadataConsumerForMetrics {
  MetadataConsumerForMetrics NO_OP =
      new MetadataConsumerForMetrics() {
        @Override
        public void accumulate(FileArtifactValue metadata) {}

        @Override
        public void accumulate(TreeArtifactValue treeArtifactValue) {}

        @Override
        public void accumulate(FilesetOutputTree filesetOutput) {}
      };

  void accumulate(FileArtifactValue metadata);

  void accumulate(TreeArtifactValue treeArtifactValue);

  void accumulate(FilesetOutputTree filesetOutput);

  /** Accumulates file metadata for later export to a {@link ArtifactMetrics.FilesMetric} object. */
  class FilesMetricConsumer implements MetadataConsumerForMetrics {
    private final AtomicLong size = new AtomicLong();
    private final AtomicInteger count = new AtomicInteger();

    @Override
    public void accumulate(FileArtifactValue metadata) {
      // Exclude directories (might throw in future) and symlinks (duplicate data). In practice,
      // most symlinks' metadata is that of their target, so they still get duplicated.
      if (metadata.getType() == FileStateType.REGULAR_FILE) {
        size.addAndGet(metadata.getSize());
        count.incrementAndGet();
      }
    }

    @Override
    public void accumulate(TreeArtifactValue treeArtifactValue) {
      long totalChildBytes = treeArtifactValue.getTotalChildBytes();
      size.addAndGet(totalChildBytes);
      if (totalChildBytes > 0) {
        // Skip omitted/missing tree artifacts: they will throw here.
        count.addAndGet(treeArtifactValue.getChildren().size());
      }
    }

    @Override
    public void accumulate(FilesetOutputTree filesetOutput) {
      // This is a bit of a fudge: we include the symlinks as a count, but don't count their
      // targets' sizes, because (a) plumbing the data is hard, (b) it would double-count symlinks
      // to output files, and (c) it's not even uniquely generated content for input files.
      count.addAndGet(filesetOutput.size());
    }

    @ThreadSafety.ThreadSafe
    public void mergeIn(FilesMetricConsumer otherConsumer) {
      this.size.addAndGet(otherConsumer.size.get());
      this.count.addAndGet(otherConsumer.count.get());
    }

    ArtifactMetrics.FilesMetric toFilesMetricAndReset() {
      return ArtifactMetrics.FilesMetric.newBuilder()
          .setSizeInBytes(size.getAndSet(0L))
          .setCount(count.getAndSet(0))
          .build();
    }

    void reset() {
      size.set(0L);
      count.set(0);
    }
  }
}
