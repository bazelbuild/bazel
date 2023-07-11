// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.analysis.InconsistentNullConfigException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.skyframe.SkyFunctionException;

/** Exceptions thrown by {@link ConfiguredTargetFunction}. */
public final class ConfiguredTargetEvaluationExceptions {
  /**
   * {@link ConfiguredTargetFunction#compute} exception that has already had its error reported to
   * the user. Callers (like {@link com.google.devtools.build.lib.buildtool.BuildTool}) won't also
   * report the error.
   */
  public static class ReportedException extends SkyFunctionException {
    ReportedException(ConfiguredValueCreationException e) {
      super(withoutMessage(e), Transience.PERSISTENT);
    }

    /** Clones a {@link ConfiguredValueCreationException} with its {@code message} field removed. */
    private static ConfiguredValueCreationException withoutMessage(
        ConfiguredValueCreationException orig) {
      return new ConfiguredValueCreationException(
          orig.getLocation(),
          "",
          /* label= */ null,
          orig.getConfiguration(),
          orig.getRootCauses(),
          orig.getDetailedExitCode());
    }
  }

  /**
   * {@link ConfiguredTargetFunction#compute} exception that has not had its error reported to the
   * user. Callers (like {@link com.google.devtools.build.lib.buildtool.BuildTool}) are responsible
   * for reporting the error.
   */
  public static class UnreportedException extends SkyFunctionException {
    UnreportedException(ConfiguredValueCreationException e) {
      super(e, Transience.PERSISTENT);
    }
  }

  /** A dependency error that should be caught and rethrown by the parent with more context. */
  static class DependencyException extends SkyFunctionException {
    enum Kind {
      INCONSISTENT_NULL_CONFIG,
      NO_SUCH_THING
    }

    private final Kind kind;

    Kind kind() {
      return kind;
    }

    InconsistentNullConfigException inconsistentNullConfig() {
      return (InconsistentNullConfigException) getCause();
    }

    NoSuchThingException noSuchThing() {
      return (NoSuchThingException) getCause();
    }

    DependencyException(InconsistentNullConfigException e) {
      super(e, Transience.PERSISTENT);
      this.kind = Kind.INCONSISTENT_NULL_CONFIG;
    }

    DependencyException(NoSuchThingException e) {
      super(e, Transience.PERSISTENT);
      this.kind = Kind.NO_SUCH_THING;
    }
  }

  private ConfiguredTargetEvaluationExceptions() {}
}
