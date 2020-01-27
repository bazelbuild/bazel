// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.util;

import static java.nio.charset.StandardCharsets.UTF_8;

/**
 * Abstract class for string indexers.
 */
abstract class AbstractIndexer implements StringIndexer {

  /**
   * Conversion from String to byte[].
   */
  protected static byte[] string2bytes(String string) {
    return string.getBytes(UTF_8);
  }

  /**
   * Conversion from byte[] to String.
   */
  protected static String bytes2string(byte[] bytes) {
    return new String(bytes, UTF_8);
  }
}
