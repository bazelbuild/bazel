/* Copyright 2017 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

package org.brotli.wrapper.common;

import java.nio.ByteBuffer;

/**
 * JNI wrapper for brotli common.
 */
class CommonJNI {
  static native boolean nativeSetDictionaryData(ByteBuffer data);
}
