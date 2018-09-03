// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylarkdebug.server;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.ContinueExecutionResponse;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.DebugEvent;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.Error;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.EvaluateResponse;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.Frame;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.GetChildrenResponse;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.ListFramesResponse;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.PauseThreadResponse;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.PausedThread;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.Scope;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.SetBreakpointsResponse;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.StartDebuggingResponse;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.ThreadContinuedEvent;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.ThreadPausedEvent;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.Value;
import com.google.devtools.build.lib.syntax.DebugFrame;
import com.google.devtools.build.lib.syntax.Debuggable.Stepping;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Helper class for constructing event or response protos to be sent from the debug server to a
 * debugger client.
 */
final class DebugEventHelper {
  private DebugEventHelper() {}

  private static final long NO_SEQUENCE_NUMBER = 0;

  static DebugEvent error(String message) {
    return error(NO_SEQUENCE_NUMBER, message);
  }

  static DebugEvent error(long sequenceNumber, String message) {
    return DebugEvent.newBuilder()
        .setSequenceNumber(sequenceNumber)
        .setError(Error.newBuilder().setMessage(message))
        .build();
  }

  static DebugEvent setBreakpointsResponse(long sequenceNumber) {
    return DebugEvent.newBuilder()
        .setSequenceNumber(sequenceNumber)
        .setSetBreakpoints(SetBreakpointsResponse.newBuilder())
        .build();
  }

  static DebugEvent continueExecutionResponse(long sequenceNumber) {
    return DebugEvent.newBuilder()
        .setSequenceNumber(sequenceNumber)
        .setContinueExecution(ContinueExecutionResponse.newBuilder())
        .build();
  }

  static DebugEvent evaluateResponse(long sequenceNumber, Value value) {
    return DebugEvent.newBuilder()
        .setSequenceNumber(sequenceNumber)
        .setEvaluate(EvaluateResponse.newBuilder().setResult(value))
        .build();
  }

  static DebugEvent listFramesResponse(long sequenceNumber, Collection<Frame> frames) {
    return DebugEvent.newBuilder()
        .setSequenceNumber(sequenceNumber)
        .setListFrames(ListFramesResponse.newBuilder().addAllFrame(frames))
        .build();
  }

  static DebugEvent startDebuggingResponse(long sequenceNumber) {
    return DebugEvent.newBuilder()
        .setSequenceNumber(sequenceNumber)
        .setStartDebugging(StartDebuggingResponse.newBuilder())
        .build();
  }

  static DebugEvent pauseThreadResponse(long sequenceNumber) {
    return DebugEvent.newBuilder()
        .setSequenceNumber(sequenceNumber)
        .setPauseThread(PauseThreadResponse.newBuilder())
        .build();
  }

  static DebugEvent getChildrenResponse(long sequenceNumber, Collection<Value> children) {
    return DebugEvent.newBuilder()
        .setSequenceNumber(sequenceNumber)
        .setGetChildren(GetChildrenResponse.newBuilder().addAllChildren(children))
        .build();
  }

  static DebugEvent threadPausedEvent(PausedThread thread) {
    return DebugEvent.newBuilder()
        .setThreadPaused(ThreadPausedEvent.newBuilder().setThread(thread))
        .build();
  }

  static DebugEvent threadContinuedEvent(long threadId) {
    return DebugEvent.newBuilder()
        .setThreadContinued(ThreadContinuedEvent.newBuilder().setThreadId(threadId))
        .build();
  }

  @Nullable
  static SkylarkDebuggingProtos.Location getLocationProto(@Nullable Location location) {
    if (location == null) {
      return null;
    }
    Location.LineAndColumn lineAndColumn = location.getStartLineAndColumn();
    if (lineAndColumn == null) {
      return null;
    }
    return SkylarkDebuggingProtos.Location.newBuilder()
        .setLineNumber(lineAndColumn.getLine())
        .setColumnNumber(lineAndColumn.getColumn())
        .setPath(location.getPath().getPathString())
        .build();
  }

  static SkylarkDebuggingProtos.Frame getFrameProto(ThreadObjectMap objectMap, DebugFrame frame) {
    SkylarkDebuggingProtos.Frame.Builder builder =
        SkylarkDebuggingProtos.Frame.newBuilder()
            .setFunctionName(frame.functionName())
            .addAllScope(getScopes(objectMap, frame));
    if (frame.location() != null) {
      builder.setLocation(getLocationProto(frame.location()));
    }
    return builder.build();
  }

  private static ImmutableList<Scope> getScopes(ThreadObjectMap objectMap, DebugFrame frame) {
    ImmutableMap<String, Object> localVars = frame.lexicalFrameBindings();
    if (localVars.isEmpty()) {
      return ImmutableList.of(getScope(objectMap, "global", frame.globalBindings()));
    }
    Map<String, Object> globalVars = new LinkedHashMap<>(frame.globalBindings());
    // remove shadowed bindings
    localVars.keySet().forEach(globalVars::remove);

    return ImmutableList.of(
        getScope(objectMap, "local", localVars), getScope(objectMap, "global", globalVars));
  }

  private static SkylarkDebuggingProtos.Scope getScope(
      ThreadObjectMap objectMap, String name, Map<String, Object> bindings) {
    SkylarkDebuggingProtos.Scope.Builder builder =
        SkylarkDebuggingProtos.Scope.newBuilder().setName(name);
    bindings.forEach(
        (s, o) -> builder.addBinding(DebuggerSerialization.getValueProto(objectMap, s, o)));
    return builder.build();
  }

  static Stepping convertSteppingEnum(SkylarkDebuggingProtos.Stepping stepping) {
    switch (stepping) {
      case INTO:
        return Stepping.INTO;
      case OUT:
        return Stepping.OUT;
      case OVER:
        return Stepping.OVER;
      case NONE:
        return Stepping.NONE;
      case UNRECOGNIZED:
        // fall through to exception
    }
    throw new IllegalArgumentException("Unsupported stepping type");
  }
}
