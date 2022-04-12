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
package com.google.devtools.build.lib.remote.common;

import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.devtools.build.lib.actions.Spawn;
import javax.annotation.Nullable;

/** A {@link RemoteActionExecutionContext} implementation */
public class SimpleRemoteActionExecutionContext implements RemoteActionExecutionContext {

  private final Type type;
  private final Spawn spawn;
  private final RequestMetadata requestMetadata;
  private final NetworkTime networkTime;

  public SimpleRemoteActionExecutionContext(
      Type type, Spawn spawn, RequestMetadata requestMetadata, NetworkTime networkTime) {
    this.type = type;
    this.spawn = spawn;
    this.requestMetadata = requestMetadata;
    this.networkTime = networkTime;
  }

  @Override
  public Type getType() {
    return type;
  }

  @Nullable
  @Override
  public Spawn getSpawn() {
    return spawn;
  }

  @Override
  public RequestMetadata getRequestMetadata() {
    return requestMetadata;
  }

  @Override
  public NetworkTime getNetworkTime() {
    return networkTime;
  }
}
