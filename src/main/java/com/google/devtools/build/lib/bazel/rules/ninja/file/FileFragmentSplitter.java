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
//

package com.google.devtools.build.lib.bazel.rules.ninja.file;

import com.google.common.collect.Lists;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.concurrent.Callable;

/**
 * Task for splitting the contents of the {@link FileFragment} (with underlying {@link ByteBuffer}).
 * Intended to be called in parallel for the fragments of the {@link ByteBuffer} for lexing the
 * contents into independent logical declarations.
 *
 * <p>{@link ParallelFileProcessing}
 */
public class FileFragmentSplitter implements Callable<List<FileFragment>> {
  private final FileFragment fileFragment;
  private final DeclarationConsumer consumer;

  /**
   * @param fileFragment {@link FileFragment}, fragment of which should be splitted
   * @param consumer declaration consumer
   */
  public FileFragmentSplitter(FileFragment fileFragment, DeclarationConsumer consumer) {
    this.fileFragment = fileFragment;
    this.consumer = consumer;
  }

  /**
   * Returns the list of {@link FileFragment} - fragments on the bounds of the current fragment,
   * which should be potentially merged with neighbor fragments.
   *
   * <p>Combined list of {@link FileFragment} is passed to {@link DeclarationAssembler} for merging.
   */
  @Override
  public List<FileFragment> call() throws Exception {
    List<FileFragment> fragments = Lists.newArrayList();
    int start = 0;
    while (true) {
      int end = NinjaSeparatorFinder.findNextSeparator(fileFragment, start, -1);
      if (end < 0) {
        break;
      }
      FileFragment fragment = fileFragment.subFragment(start, end + 1);
      if (start > 0) {
        consumer.declaration(fragment);
      } else {
        fragments.add(fragment);
      }
      start = end + 1;
    }
    // There is always at least one byte at the bounds of the fragment.
    FileFragment lastFragment = fileFragment.subFragment(start, fileFragment.length());
    fragments.add(lastFragment);
    return fragments;
  }
}
