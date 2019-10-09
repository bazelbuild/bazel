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

/**
 * Generic callback interface to accept tokens from buffer fragments tokenizer
 * {@link BufferTokenizer}.
 */
public interface TokenAndFragmentsConsumer extends TokenConsumer {

  /**
   * Accepts the fragment of token
   * @param offset the character offset of the underlying buffer of sequence in the file
   * @param sequence the fragment char sequence, also containing information about the position
   * of the sequence in the underlying buffer
   */
  void fragment(int offset, ArrayViewCharSequence sequence);
}
