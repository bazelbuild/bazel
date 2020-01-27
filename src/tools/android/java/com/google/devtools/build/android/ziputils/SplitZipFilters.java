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
package com.google.devtools.build.android.ziputils;

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableSet;
import java.io.FileInputStream;
import java.io.IOException;

/**
 * Custom filters intended for {@link SplitZip#setInputFilter}.
 */
public class SplitZipFilters {

  /**
   * Returns a predicate that returns true for filenames contained in the given zip file.
   */
  public static Predicate<String> entriesIn(String filterZip) throws IOException {
    // Aggregate filenames into a set so Predicates.in is efficient
    ImmutableSet.Builder<String> filenames = ImmutableSet.builder();
    @SuppressWarnings("resource") // ZipIn takes ownership but isn't Closable
    ZipIn zip = new ZipIn(new FileInputStream(filterZip).getChannel(), filterZip);
    for (DirectoryEntry entry : zip.centralDirectory().list()) {
      filenames.add(entry.getFilename());
    }
    return Predicates.in(filenames.build());
  }

  private SplitZipFilters() {
  }
}
