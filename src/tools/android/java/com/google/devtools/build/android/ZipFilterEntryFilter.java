// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import com.google.common.collect.Multimap;
import com.google.devtools.build.android.ZipFilterAction.HashMismatchCheckMode;
import com.google.devtools.build.singlejar.ZipEntryFilter;
import java.io.IOException;
import java.util.Map;

/**
 * A {@link ZipEntryFilter} for use with {@link ZipFilterAction}. Filters out entries that match the
 * provided regular expression or are in the list to omit.
 */
class ZipFilterEntryFilter implements ZipEntryFilter {

  private final String explicitFilter;
  private final Multimap<String, Long> entriesToOmit;
  private final Map<String, Long> inputEntries;
  private final HashMismatchCheckMode hashMismatchCheckMode;

  /**
   * Creates a new filter.
   *
   * @param explicitFilter a regular expression to match against entry filenames
   * @param entriesToOmit a map of filename and CRC-32 of entries to omit
   * @param inputEntries a map of filename and CRC-32 of entries in the input Zip file
   * @param hashMismatchCheckMode ignore, warn or error out for content hash mismatches.
   */
  public ZipFilterEntryFilter(
      String explicitFilter,
      Multimap<String, Long> entriesToOmit,
      Map<String, Long> inputEntries,
      HashMismatchCheckMode hashMismatchCheckMode) {
    this.explicitFilter = explicitFilter;
    this.entriesToOmit = entriesToOmit;
    this.inputEntries = inputEntries;
    this.hashMismatchCheckMode = hashMismatchCheckMode;
  }

  @Override
  public void accept(String filename, StrategyCallback callback) throws IOException {
    if (filename.matches(explicitFilter)) {
      callback.skip();
    } else if (entriesToOmit.containsKey(filename)) {
      if (hashMismatchCheckMode == HashMismatchCheckMode.IGNORE) {
        callback.skip();
      } else {
        Long entryCrc = inputEntries.get(filename);
        if (entriesToOmit.containsEntry(filename, entryCrc)) {
          callback.skip();
        } else {
          if (hashMismatchCheckMode == HashMismatchCheckMode.ERROR) {
            throw new IllegalStateException(
                String.format(
                    "Requested to filter entries of name "
                        + "'%s'; name matches but the hash does not. Aborting",
                    filename));
          } else {
            System.out.printf(
                "\u001b[35mWARNING:\u001b[0m Requested to filter entries of name "
                    + "'%s'; name matches but the hash does not. Copying anyway.\n",
                filename);
            callback.copy(null);
          }
        }
      }
    } else {
      callback.copy(null);
    }
  }
}
