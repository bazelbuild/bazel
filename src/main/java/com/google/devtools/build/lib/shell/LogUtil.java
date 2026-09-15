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
package com.google.devtools.build.lib.shell;

import java.nio.charset.StandardCharsets;

/**
 * Utilities for logging.
 */
class LogUtil {

  private LogUtil() {}

  private final static int TRUNCATE_STRINGS_AT = 150;

  /**
   * Make a string out of a byte array, and truncate it to a reasonable length.
   * Useful for preventing logs from becoming excessively large.
   */
  static String toTruncatedString(final byte[] bytes) {
    if(bytes == null || bytes.length == 0) {
      return "";
    }
    /*
     * Yes, we'll use the platform encoding here, and this is one of the rare
     * cases where it makes sense. You want the logs to be encoded so that
     * your platform tools (vi, emacs, cat) can render them, don't you?
     * In practice, this means ISO-8859-1 or UTF-8, I guess.
     */
    try {
      if (bytes.length > TRUNCATE_STRINGS_AT) {
        return new String(bytes, 0, TRUNCATE_STRINGS_AT, StandardCharsets.UTF_8)
          + "[... truncated. original size was " + bytes.length + " bytes.]";
      }
      return new String(bytes);
    } catch (Exception e) {
      /*
       * In case encoding a binary string doesn't work for some reason, we
       * don't want to bring a logging server down - do we? So we're paranoid.
       */
      return "IOUtil.toTruncatedString: " + e.getMessage();
    }
  }

}
