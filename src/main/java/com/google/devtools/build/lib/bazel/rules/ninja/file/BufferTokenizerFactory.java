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

import java.nio.ByteBuffer;
import java.util.concurrent.Callable;

/**
 * Factor interface for creating the buffer tokenizing task.
 *
 * {@link BufferTokenizer}
 */
public interface BufferTokenizerFactory {
  /**
   * Creates the new task for tokenizing the passed buffer fragment.
   *  @param buffer buffer, fragment of which should be tokenized
   * @param offset offset of this buffer (sum of the number of characters in all previous buffers,
   * read from this file)
   * @param startIncl start index of the buffer fragment, inclusive
   * @param endExcl end index of the buffer fragment, exclusive, or the size of the buffer,
   */
  Callable<Void> create(ByteBuffer buffer, int offset, int startIncl, int endExcl);
}
