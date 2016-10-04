// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.dexer;

import java.util.zip.ZipEntry;

/**
 * Structured pair of file metadata encoded as {@link ZipEntry} and {@code byte[]} file content.
 * Typically this class is used to represent an entry in a zip file.
 */
class ZipEntryContent {

  private final ZipEntry entry;
  private final byte[] content;

  public ZipEntryContent(ZipEntry entry, byte[] content) {
    this.entry = entry;
    this.content = content;
  }

  public ZipEntry getEntry() {
    return entry;
  }

  public byte[] getContent() {
    return content;
  }
}
