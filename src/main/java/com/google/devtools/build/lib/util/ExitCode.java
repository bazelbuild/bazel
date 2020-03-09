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

package com.google.devtools.build.lib.util;

import com.google.common.base.Objects;
import java.util.Collection;
import java.util.HashMap;
import javax.annotation.Nullable;

/**
 *  <p>Anything marked FAILURE is generally from a problem with the source code
 *  under consideration.  In these cases, a re-run in an identical client should
 *  produce an identical return code all things being constant.
 *
 *  <p>Anything marked as an ERROR is generally a problem unrelated to the
 *  source code itself.  It is either something wrong with the user's command
 *  line or the user's machine or environment.
 *
 *  <p>Note that these exit codes should be kept consistent with the codes
 *  returned by Blaze's launcher in //devtools/blaze/main:blaze.cc
 *  Blaze exit codes should be consistently classified as permanent vs.
 *  transient (i.e. retriable) vs. unknown transient/permanent because users,
 *  in particular infrastructure users, will use the exit code to decide whether
 *  the request should be retried or not.
 */
public class ExitCode {
  // Tracks all exit codes defined here and elsewhere in Bazel.
  private static final HashMap<Integer, ExitCode> exitCodeRegistry = new HashMap<>();

  public static final ExitCode SUCCESS = ExitCode.create(0, "SUCCESS");
  public static final ExitCode BUILD_FAILURE = ExitCode.create(1, "BUILD_FAILURE");
  public static final ExitCode PARSING_FAILURE = ExitCode.createUnregistered(1, "PARSING_FAILURE");
  public static final ExitCode COMMAND_LINE_ERROR = ExitCode.create(2, "COMMAND_LINE_ERROR");
  public static final ExitCode TESTS_FAILED = ExitCode.create(3, "TESTS_FAILED");
  public static final ExitCode PARTIAL_ANALYSIS_FAILURE =
      ExitCode.createUnregistered(3, "PARTIAL_ANALYSIS_FAILURE");
  public static final ExitCode NO_TESTS_FOUND = ExitCode.create(4, "NO_TESTS_FOUND");
  public static final ExitCode RUN_FAILURE = ExitCode.create(6, "RUN_FAILURE");
  public static final ExitCode ANALYSIS_FAILURE = ExitCode.create(7, "ANALYSIS_FAILURE");
  public static final ExitCode INTERRUPTED = ExitCode.create(8, "INTERRUPTED");
  public static final ExitCode LOCK_HELD_NOBLOCK_FOR_LOCK =
      ExitCode.create(9, "LOCK_HELD_NOBLOCK_FOR_LOCK");

  public static final ExitCode REMOTE_ENVIRONMENTAL_ERROR =
      ExitCode.createInfrastructureFailure(32, "REMOTE_ENVIRONMENTAL_ERROR");
  public static final ExitCode OOM_ERROR = ExitCode.createInfrastructureFailure(33, "OOM_ERROR");

  public static final ExitCode REMOTE_ERROR =
      ExitCode.createInfrastructureFailure(34, "REMOTE_ERROR");
  public static final ExitCode LOCAL_ENVIRONMENTAL_ERROR =
      ExitCode.createInfrastructureFailure(36, "LOCAL_ENVIRONMENTAL_ERROR");
  public static final ExitCode BLAZE_INTERNAL_ERROR =
      ExitCode.createInfrastructureFailure(37, "BLAZE_INTERNAL_ERROR");
  public static final ExitCode TRANSIENT_BUILD_EVENT_SERVICE_UPLOAD_ERROR =
      ExitCode.createInfrastructureFailure(38, "PUBLISH_ERROR");
  public static final ExitCode PERSISTENT_BUILD_EVENT_SERVICE_UPLOAD_ERROR =
      ExitCode.create(45, "PERSISTENT_BUILD_EVENT_SERVICE_UPLOAD_ERROR");

  public static final ExitCode RESERVED = ExitCode.createInfrastructureFailure(40, "RESERVED");

  /*
    exit codes [50..60] and 253 are reserved for site specific wrappers to Bazel.
   */

  /**
   * Creates and returns an ExitCode.  Requires a unique exit code number.
   *
   * @param code the int value for this exit code
   * @param name a human-readable description
   */
  public static ExitCode create(int code, String name) {
    return new ExitCode(code, name, /*infrastructureFailure=*/false, /*register=*/true);
  }

  /**
   * Creates and returns an ExitCode that represents an infrastructure failure.
   *
   * @param code the int value for this exit code
   * @param name a human-readable description
   */
  public static ExitCode createInfrastructureFailure(int code, String name) {
    return new ExitCode(code, name, /*infrastructureFailure=*/true, /*register=*/true);
  }

  /**
   * Creates and returns an ExitCode that has the same numeric code as another ExitCode. This is to
   * allow the duplicate error codes listed above to be registered, but is private to prevent other
   * users from creating duplicate error codes in the future.
   *
   * @param code the int value for this exit code
   * @param name a human-readable description
   */
  private static ExitCode createUnregistered(int code, String name) {
    return new ExitCode(code, name, /*infrastructureFailure=*/false, /*register=*/false);
  }

  /**
   * Add the given exit code to the registry.
   *
   * @param exitCode the exit code to register
   * @throws IllegalStateException if the numeric exit code is already in the registry.
   */
  private static void register(ExitCode exitCode) {
    synchronized (exitCodeRegistry) {
      int codeNum = exitCode.getNumericExitCode();
      if (exitCodeRegistry.containsKey(codeNum)) {
        throw new IllegalStateException(
            "Exit code " + codeNum + " (" + exitCode.name + ") already registered");
      }
      exitCodeRegistry.put(codeNum, exitCode);
    }
  }

  /**
   * Returns all registered ExitCodes.
   */
  public static Collection<ExitCode> values() {
    synchronized (exitCodeRegistry) {
      return exitCodeRegistry.values();
    }
  }

  /**
   * Returns a registered {@link ExitCode} with the given {@code code}.
   *
   * <p>Note that there *are* unregistered ExitCodes. This will never return them.
   */
  @Nullable
  static ExitCode forCode(int code) {
    synchronized (exitCodeRegistry) {
      return exitCodeRegistry.get(code);
    }
  }

  private final int numericExitCode;
  private final String name;
  private final boolean infrastructureFailure;

  /**
   * Whenever a new exit code is created, it is registered (to prevent exit codes with identical
   * numeric codes from being created).  However, there are some exit codes in this file that have
   * duplicate numeric codes, so these are not registered.
   */
  private ExitCode(int exitCode, String name, boolean infrastructureFailure, boolean register) {
    this.numericExitCode = exitCode;
    this.name = name;
    this.infrastructureFailure = infrastructureFailure;
    if (register) {
      ExitCode.register(this);
    }
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(numericExitCode, name, infrastructureFailure);
  }

  @Override
  public boolean equals(Object object) {
    if (object instanceof ExitCode) {
      ExitCode that = (ExitCode) object;
      return this.numericExitCode == that.numericExitCode
          && this.name.equals(that.name)
          && this.infrastructureFailure == that.infrastructureFailure;
    }
    return false;
  }

  /**
   * Returns the human-readable name for this exit code.  Not guaranteed to be stable, use the
   * numeric exit code for that.
   */
  @Override
  public String toString() {
    return name;
  }

  /**
   * Returns the error's int value.
   */
  public int getNumericExitCode() {
    return numericExitCode;
  }

  /**
   * Returns the human-readable name.
   */
  public String name() {
    return name;
  }

  /**
   * Returns true if the current exit code represents a failure of Blaze infrastructure,
   * vs. a build failure.
   */
  public boolean isInfrastructureFailure() {
    return infrastructureFailure;
  }
}
