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

/** Generic interface to accept declarations from {@link ParallelFileProcessing} */
public interface DeclarationConsumer {

  /**
   * Accepts a declaration for further processing / parsing.
   *
   * @param fragment fragment of {@link java.nio.ByteBuffer}
   * @throws GenericParsingException if declaration processing discovered the wrong syntax
   */
  void declaration(ByteBufferFragment fragment) throws GenericParsingException;
}
