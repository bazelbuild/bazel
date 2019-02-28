// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote.blobstore.http;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.timeout.IdleState;
import io.netty.handler.timeout.IdleStateEvent;
import io.netty.handler.timeout.IdleStateHandler;
import io.netty.handler.timeout.TimeoutException;
import java.util.concurrent.TimeUnit;

public class IdleTimeoutHandler extends IdleStateHandler {
  private TimeoutException timeoutException;
  private boolean closed;

  public IdleTimeoutHandler(long timeoutSeconds, TimeoutException timeoutException) {
    super(0, 0, timeoutSeconds, TimeUnit.SECONDS);
    this.timeoutException = timeoutException;
  }

  @Override
  protected final void channelIdle(ChannelHandlerContext ctx, IdleStateEvent evt) throws Exception {
    assert evt.state() == IdleState.ALL_IDLE;
    if (!closed) {
      ctx.fireExceptionCaught(timeoutException);
      ctx.close();
      closed = true;
    }
  }
}
