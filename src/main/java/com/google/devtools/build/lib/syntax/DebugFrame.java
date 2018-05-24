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

package com.google.devtools.build.lib.syntax;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.Location;
import javax.annotation.Nullable;

/** The information about a single frame in a thread's stack trace relevant to the debugger. */
@AutoValue
public abstract class DebugFrame {
  /** The source location where the frame is currently paused. */
  @Nullable
  public abstract Location location();

  /** The name of the function that this frame represents. */
  public abstract String functionName();

  /**
   * The local bindings associated with the current lexical frame. For the outer-most scope this
   * will be empty.
   */
  public abstract ImmutableMap<String, Object> lexicalFrameBindings();

  /** The global vars and builtins for this frame. May be shadowed by the lexical frame bindings. */
  public abstract ImmutableMap<String, Object> globalBindings();

  public static Builder builder() {
    return new AutoValue_DebugFrame.Builder().setLexicalFrameBindings(ImmutableMap.of());
  }

  /** Builder class for {@link DebugFrame}. */
  @AutoValue.Builder
  public abstract static class Builder {
    public abstract Builder setLocation(@Nullable Location location);

    public abstract Builder setFunctionName(String functionName);

    public abstract Builder setLexicalFrameBindings(ImmutableMap<String, Object> bindings);

    public abstract Builder setGlobalBindings(ImmutableMap<String, Object> bindings);

    public abstract DebugFrame build();
  }
}
