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
package com.google.devtools.common.options;

/**
 * On top of categorizing options by their intended purpose, these tags should identify options that
 * are either not supported or are intended to break old behavior.
 */
public enum OptionMetadataTag {
  /**
   * This option triggers an experimental feature with no guarantees of functionality.
   *
   * <p>Note: this is separate from UNDOCUMENTED flags, which are flags we don't want listed and
   * shouldn't be widely used. Experimental flags should probably also be undocumented, but not all
   * undocumented flags should be labeled experimental.
   */
  EXPERIMENTAL(0),

  /**
   * This option triggers a backwards-incompatible change. It will be off by default when the option
   * is first introduced, and later switched on by default on a major Blaze release. Use this option
   * to test your migration readiness or get early access to the feature. The option may be
   * deprecated some time after the feature's release.
   */
  INCOMPATIBLE_CHANGE(1),

  /**
   * This flag is deprecated. It might either no longer have any effect, or might no longer be
   * supported.
   */
  DEPRECATED(2),

  /**
   * These are flags that should never be set by a user. This tag is used to make sure that options
   * that form the protocol between the client and the server are not logged.
   *
   * <p>These should be in category {@code OptionDocumentationCategory.UNDOCUMENTED}.
   */
  HIDDEN(3),

  /**
   * Options which are INTERNAL are not recognized by the parser at all, and so cannot be used as
   * flags.
   *
   * <p>These should be in category {@code OptionDocumentationCategory.UNDOCUMENTED}.
   */
  INTERNAL(4),

  /**
   * Options that are triggered by --all_incompatible_changes.
   *
   * <p>These must also be labelled {@link OptionMetadataTag#INCOMPATIBLE_CHANGE} and have the
   * prefix --incompatible_. Note that the option name prefix is also a triggering case for the
   * --all_incompatible_changes expansion, and so all options that start with the "incompatible_"
   * prefix must have this tag.
   */
  TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES(5);

  private final int value;

  OptionMetadataTag(int value) {
    this.value = value;
  }

  public int getValue() {
    return value;
  }
}
