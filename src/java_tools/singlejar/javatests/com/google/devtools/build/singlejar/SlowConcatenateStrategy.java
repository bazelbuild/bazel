// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.singlejar;


import com.google.devtools.build.singlejar.ZipEntryFilter.CustomMergeStrategy;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import javax.annotation.concurrent.NotThreadSafe;

/**
 * A strategy that merges a set of files by concatenating them. It inserts no
 * additional characters and copies bytes one by one. Used for testing.
 */
@NotThreadSafe
final class SlowConcatenateStrategy implements CustomMergeStrategy {

  @Override
  public void merge(InputStream in, OutputStream out) throws IOException {
    int nextByte;
    while ((nextByte = in.read()) != -1) {
      out.write(nextByte);
    }
  }

  @Override
  public void finish(OutputStream out) {
    // No need to do anything. All the data was already written.
  }
}
