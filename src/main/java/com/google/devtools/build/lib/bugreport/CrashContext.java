// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bugreport;

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.List;
import javax.annotation.Nullable;

/** Context describing when a {@link Crash} occurred and how it should be handled. */
public final class CrashContext {

  /**
   * Creates a {@link CrashContext} that instructs {@link BugReporter} to halt the JVM when handling
   * a crash.
   *
   * <p>This should only be used when it is not feasible to conduct an orderly shutdown, for example
   * a crash in an async thread.
   */
  public static CrashContext halt() {
    return new CrashContext(/* haltJvm= */ true, /* returnIfCrashInProgress= */ false);
  }

  /**
   * Creates a {@link CrashContext} that instructs {@link BugReporter} <em>not</em> to halt the JVM
   * when handling a crash.
   *
   * <p>The caller is responsible for terminating the server with an appropriate exit code.
   */
  public static CrashContext keepAlive() {
    return new CrashContext(/* haltJvm= */ false, /* returnIfCrashInProgress= */ false);
  }

  /**
   * Creates a {@link CrashContext} that instructs {@link BugReporter} to halt the JVM when handling
   * a crash if there is no other crash in progress, and return otherwise.
   *
   * <p>This should only be used when it is not feasible to conduct an orderly shutdown, for example
   * a crash in an async thread, and where that async thread must make progress while another crash
   * is already shutting down the {@code BlazeRuntime}. This can prevent deadlocks during shutdown.
   */
  public static CrashContext haltOrReturnIfCrashInProgress() {
    return new CrashContext(/* haltJvm= */ true, /* returnIfCrashInProgress= */ true);
  }

  private final boolean haltJvm;
  private ImmutableList<String> args = ImmutableList.of();
  private boolean sendBugReport = true;
  private String extraOomInfo = "";
  @Nullable private String heapDumpPath = null;
  private EventHandler eventHandler =
      event -> System.err.println(event.getKind() + ": " + event.getMessage());
  private final boolean returnIfCrashInProgress;

  private CrashContext(boolean haltJvm, boolean returnIfCrashInProgress) {
    this.haltJvm = haltJvm;
    this.returnIfCrashInProgress = returnIfCrashInProgress;
  }

  /** Sets the arguments that {@link BugReporter} should include with the bug report. */
  @CanIgnoreReturnValue
  public CrashContext withArgs(String... args) {
    this.args = ImmutableList.copyOf(args);
    return this;
  }

  /** Sets the arguments that {@link BugReporter} should include with the bug report. */
  @CanIgnoreReturnValue
  public CrashContext withArgs(List<String> args) {
    this.args = ImmutableList.copyOf(args);
    return this;
  }

  /** Disables bug reporting. */
  @CanIgnoreReturnValue
  public CrashContext withoutBugReport() {
    sendBugReport = false;
    return this;
  }

  /**
   * Sets a custom additional message that should be including when handling an {@link
   * OutOfMemoryError}.
   */
  @CanIgnoreReturnValue
  public CrashContext withExtraOomInfo(String extraOomInfo) {
    this.extraOomInfo = extraOomInfo;
    return this;
  }

  /**
   * Sets the path at which to write a heap dump when handling {@link OutOfMemoryError}.
   *
   * <p>The path <em>must</em> end in {@code .hprof} for the heap dump to succeed.
   *
   * <p>If not called, there will be no heap dump.
   */
  public void setHeapDumpPath(String heapDumpPath) {
    this.heapDumpPath = heapDumpPath;
  }

  /**
   * Sets the {@link EventHandler} that should be notified about the {@link EventKind#FATAL} crash
   * event.
   *
   * <p>If this method is not called, the event is printed to {@link System#err}.
   */
  @CanIgnoreReturnValue
  public CrashContext reportingTo(EventHandler eventHandler) {
    this.eventHandler = eventHandler;
    return this;
  }

  boolean shouldHaltJvm() {
    return haltJvm;
  }

  ImmutableList<String> getArgs() {
    return args;
  }

  boolean shouldSendBugReport() {
    return sendBugReport;
  }

  String getExtraOomInfo() {
    return extraOomInfo;
  }

  @Nullable
  String getHeapDumpPath() {
    return heapDumpPath;
  }

  EventHandler getEventHandler() {
    return eventHandler;
  }

  boolean returnIfCrashInProgress() {
    return returnIfCrashInProgress;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("haltJvm", haltJvm)
        .add("args", args)
        .add("sendBugReport", sendBugReport)
        .add("extraOomInfo", extraOomInfo)
        .add("eventHandler", eventHandler)
        .toString();
  }
}
