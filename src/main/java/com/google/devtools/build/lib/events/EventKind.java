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
package com.google.devtools.build.lib.events;

import java.util.EnumSet;
import java.util.Set;

/**
 * Indicates the kind of an {@link Event}.
 */
public enum EventKind {

  /**
   * For errors that will prevent a successful, correct build.  In general, the
   * build tool will not attempt to start or continue a build if an error is
   * encountered (though the behaviour specified by --keep-going flag is a
   * counterexample).
   *
   * Errors of a more severe nature in the input, such as those which might
   * cause later passes of the analysis to fail catastrophically, should be
   * handled by throwing an exception.
   */
  ERROR,

  /**
   * For warnings of minor problems that do not affect the integrity of a
   * build.
   */
  WARNING,

  /**
   * For atemporal information that is true throughout the entire duration
   * of a build. (e.g. the number of targets found)
   */
  INFO,

  /**
   * For temporal information that changes during the duration of a build.
   * (e.g. what action is executing now)
   */
  PROGRESS,

  /**
   * For debug messages
   */
  DEBUG,

  /**
   * For progress messages (temporal information) relating to the start
   * and end of particular tasks.
   * (e.g. "Loading package foo", "Compiling bar", etc.)
   */
  START,
  FINISH,

  /**
   * For command lines of subcommands executed by the build tool (like make-dbg
   * "-v").
   */
  SUBCOMMAND,

  /**
   * Output to stdout/stderr from subprocesses.
   */
  STDOUT,
  STDERR,

  /**
   * Test result messages (similar to the INFO and ERROR, but test-specific).
   */
  PASS,
  FAIL,
  TIMEOUT,

  /**
   * For the reasoning of the dependency checker (like GNU Make "-d").
   */
  DEPCHECKER;

  // Convenient predefined EnumSets.  Clients should not mutate them!

  public static final Set<EventKind> ALL_EVENTS =
      EnumSet.allOf(EventKind.class);

  public static final Set<EventKind> OUTPUT = EnumSet.of(
      EventKind.STDOUT,
      EventKind.STDERR
      );

  public static final Set<EventKind> ERRORS = EnumSet.of(
      EventKind.ERROR,
      EventKind.FAIL,
      EventKind.TIMEOUT
      );

  public static final Set<EventKind> ERRORS_AND_WARNINGS = EnumSet.of(
      EventKind.ERROR,
      EventKind.WARNING,
      EventKind.DEBUG,
      EventKind.FAIL,
      EventKind.TIMEOUT
      );

  public static final Set<EventKind> ERRORS_WARNINGS_AND_INFO = EnumSet.of(
      EventKind.ERROR,
      EventKind.WARNING,
      EventKind.DEBUG,
      EventKind.PASS,
      EventKind.FAIL,
      EventKind.TIMEOUT,
      EventKind.INFO
      );

  public static final Set<EventKind> ERRORS_AND_OUTPUT = EnumSet.of(
      EventKind.ERROR,
      EventKind.FAIL,
      EventKind.TIMEOUT,
      EventKind.STDOUT,
      EventKind.STDERR
      );

  public static final Set<EventKind> ERRORS_AND_WARNINGS_AND_OUTPUT = EnumSet.of(
      EventKind.ERROR,
      EventKind.WARNING,
      EventKind.DEBUG,
      EventKind.FAIL,
      EventKind.TIMEOUT,
      EventKind.STDOUT,
      EventKind.STDERR
      );

  public static final Set<EventKind> ERRORS_WARNINGS_AND_INFO_AND_OUTPUT = EnumSet.of(
      EventKind.ERROR,
      EventKind.WARNING,
      EventKind.DEBUG,
      EventKind.PASS,
      EventKind.FAIL,
      EventKind.TIMEOUT,
      EventKind.INFO,
      EventKind.STDOUT,
      EventKind.STDERR
      );

}
