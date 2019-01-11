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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.remote.metrics.RemoteMetrics;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelDuplexHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelPromise;

/**
 * Collects read/write metrics of the {@link HttpBlobStore}.
 */
public class MetricsHandler extends ChannelDuplexHandler {

  private final RemoteMetrics metrics;

  public MetricsHandler(RemoteMetrics metrics) {
    this.metrics = Preconditions.checkNotNull(metrics);
  }

  @Override
  public void write(ChannelHandlerContext ctx, Object msg, ChannelPromise promise)
      throws Exception {
    if (msg instanceof ByteBuf) {
      ByteBuf b = (ByteBuf) msg;
      metrics.bytesSent(b.writableBytes());
    }
    super.write(ctx, msg, promise);
  }

  @Override
  public void channelRead(ChannelHandlerContext ctx, Object msg) throws Exception {
    if (msg instanceof ByteBuf) {
      ByteBuf b = (ByteBuf) msg;
      metrics.bytesReceived(b.readableBytes());
    }
    super.channelRead(ctx, msg);
  }
}
