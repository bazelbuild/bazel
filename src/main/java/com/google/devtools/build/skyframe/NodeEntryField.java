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
package com.google.devtools.build.skyframe;

import java.util.EnumSet;

/**
 * Fields of a {@link NodeEntry} that clients of a {@link QueryableGraph} may need. Clients may
 * specify these fields in {@link QueryableGraph#getBatchWithFieldHints} to help particular {@link
 * QueryableGraph} implementations decide how lazily to construct the returned node entries.
 */
public enum NodeEntryField {
  /** The value ({@link NodeEntry#getValueMaybeWithMetadata}) will be needed. */
  VALUE,
  /** The direct deps ({@link NodeEntry#getDirectDeps}) will be needed. */
  DIRECT_DEPS,
  /** The reverse deps ({@link NodeEntry#getReverseDeps}) will be needed. */
  REVERSE_DEPS,
  /**
   * The reverse deps as a whole will not be needed, but we may need to check the presence of a
   * reverse dep or add/delete one.
   */
  INDIVIDUAL_REVERSE_DEPS;

  public static final EnumSet<NodeEntryField> NO_FIELDS = EnumSet.noneOf(NodeEntryField.class);
  public static final EnumSet<NodeEntryField> VALUE_ONLY = EnumSet.of(VALUE);
  public static final EnumSet<NodeEntryField> NO_VALUE = EnumSet.of(DIRECT_DEPS, REVERSE_DEPS);
  public static final EnumSet<NodeEntryField> ALL_FIELDS =
      EnumSet.of(VALUE, DIRECT_DEPS, REVERSE_DEPS);
}
