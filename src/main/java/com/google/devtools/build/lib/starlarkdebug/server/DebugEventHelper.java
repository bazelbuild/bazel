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

package com.google.devtools.build.lib.starlarkdebug.server;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.ContinueExecutionResponse;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.DebugEvent;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.Error;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.EvaluateResponse;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.Frame;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.GetChildrenResponse;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.ListFramesResponse;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.PauseThreadResponse;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.PausedThread;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.Scope;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.SetBreakpointsResponse;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.StartDebuggingResponse;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.ThreadContinuedEvent;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.ThreadPausedEvent;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.Value;
import com.google.devtools.build.lib.syntax.Debug;
import com.google.devtools.build.lib.syntax.Location;
import com.google.devtools.build.lib.syntax.StarlarkFunction;
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
  static StarlarkDebuggingProtos.Location getLocationProto(@Nullable Location location) {
    if (location == null) {
      return null;
    }
    return StarlarkDebuggingProtos.Location.newBuilder()
        .setLineNumber(location.line())
        .setColumnNumber(location.column())
        .setPath(location.file())
        .build();
  }

  static StarlarkDebuggingProtos.Frame getFrameProto(ThreadObjectMap objectMap, Debug.Frame frame) {
    return StarlarkDebuggingProtos.Frame.newBuilder()
        .setFunctionName(frame.getFunction().getName())
        .addAllScope(getScopes(objectMap, frame))
        .setLocation(getLocationProto(frame.getLocation()))
        .build();
  }

  private static ImmutableList<Scope> getScopes(ThreadObjectMap objectMap, Debug.Frame frame) {
    Map<String, Object> moduleVars =
        frame.getFunction() instanceof StarlarkFunction
            ? ((StarlarkFunction) frame.getFunction()).getModule().getGlobals()
            : ImmutableMap.of();

    ImmutableMap<String, Object> localVars = frame.getLocals();
    if (localVars.isEmpty()) {
      return ImmutableList.of(getScope(objectMap, "global", moduleVars));
    }

    Map<String, Object> globalVars = new LinkedHashMap<>(moduleVars);
    // remove shadowed bindings
    localVars.keySet().forEach(globalVars::remove);

    return ImmutableList.of(
        getScope(objectMap, "local", localVars), getScope(objectMap, "global", globalVars));
  }

  private static StarlarkDebuggingProtos.Scope getScope(
      ThreadObjectMap objectMap, String name, Map<String, Object> bindings) {
    StarlarkDebuggingProtos.Scope.Builder builder =
        StarlarkDebuggingProtos.Scope.newBuilder().setName(name);
    bindings.forEach(
        (s, o) -> builder.addBinding(DebuggerSerialization.getValueProto(objectMap, s, o)));
    return builder.build();
  }

  static Debug.Stepping convertSteppingEnum(StarlarkDebuggingProtos.Stepping stepping) {
    switch (stepping) {
      case INTO:
        return Debug.Stepping.INTO;
      case OUT:
        return Debug.Stepping.OUT;
      case OVER:
        return Debug.Stepping.OVER;
      case NONE:
        return Debug.Stepping.NONE;
      case UNRECOGNIZED:
        // fall through to exception
    }
    throw new IllegalArgumentException("Unsupported stepping type");
  }
}
