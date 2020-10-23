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

package com.google.devtools.build.lib.remote.http;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.net.HttpHeaders;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.ByteBufAllocator;
import io.netty.buffer.ByteBufUtil;
import io.netty.channel.ChannelPromise;
import io.netty.channel.embedded.EmbeddedChannel;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpChunkedInput;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpRequest;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpVersion;
import java.io.ByteArrayInputStream;
import java.net.URI;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link HttpUploadHandler}. */
@RunWith(JUnit4.class)
public class HttpUploadHandlerTest {

  private static final URI CACHE_URI = URI.create("http://storage.googleapis.com:80/cache-bucket");

  /**
   * Test that uploading blobs works to both the Action Cache and the CAS. Also test that the
   * handler is reusable.
   */
  @Test
  public void uploadsShouldWork() throws Exception {
    EmbeddedChannel ch = new EmbeddedChannel(new HttpUploadHandler(null, ImmutableList.of()));
    HttpResponseStatus[] statuses =
        new HttpResponseStatus[] {
          HttpResponseStatus.OK,
          HttpResponseStatus.CREATED,
          HttpResponseStatus.ACCEPTED,
          HttpResponseStatus.NO_CONTENT
        };

    for (HttpResponseStatus status : statuses) {
      uploadsShouldWork(true, ch, status);
      uploadsShouldWork(false, ch, status);
    }
  }

  private void uploadsShouldWork(boolean casUpload, EmbeddedChannel ch, HttpResponseStatus status)
      throws Exception {
    ByteArrayInputStream data = new ByteArrayInputStream(new byte[] {1, 2, 3, 4, 5});
    ChannelPromise writePromise = ch.newPromise();
    ch.writeOneOutbound(new UploadCommand(CACHE_URI, casUpload, "abcdef", data, 5), writePromise);

    HttpRequest request = ch.readOutbound();
    assertThat(request.method()).isEqualTo(HttpMethod.PUT);
    assertThat(request.headers().get(HttpHeaders.CONNECTION))
        .isEqualTo(HttpHeaderValues.KEEP_ALIVE.toString());

    HttpChunkedInput content = ch.readOutbound();
    assertThat(content.readChunk(ByteBufAllocator.DEFAULT).content().readableBytes()).isEqualTo(5);

    FullHttpResponse response = new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, status);
    response.headers().set(HttpHeaderNames.CONNECTION, HttpHeaderValues.KEEP_ALIVE);

    ch.writeInbound(response);

    assertThat(writePromise.isDone()).isTrue();
    assertThat(ch.isOpen()).isTrue();
  }

  /** Test that the handler correctly supports http error codes i.e. 404 (NOT FOUND). */
  @Test
  public void httpErrorsAreSupported() {
    EmbeddedChannel ch = new EmbeddedChannel(new HttpUploadHandler(null, ImmutableList.of()));
    ByteArrayInputStream data = new ByteArrayInputStream(new byte[] {1, 2, 3, 4, 5});
    ChannelPromise writePromise = ch.newPromise();
    ch.writeOneOutbound(new UploadCommand(CACHE_URI, true, "abcdef", data, 5), writePromise);

    HttpRequest request = ch.readOutbound();
    assertThat(request).isInstanceOf(HttpRequest.class);
    HttpChunkedInput content = ch.readOutbound();
    assertThat(content).isInstanceOf(HttpChunkedInput.class);

    FullHttpResponse response =
        new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.FORBIDDEN);
    response.headers().set(HttpHeaders.CONNECTION, HttpHeaderValues.CLOSE);

    ch.writeInbound(response);

    assertThat(writePromise.isDone()).isTrue();
    assertThat(writePromise.cause()).isInstanceOf(HttpException.class);
    assertThat(((HttpException) writePromise.cause()).response().status())
        .isEqualTo(HttpResponseStatus.FORBIDDEN);
    assertThat(ch.isOpen()).isFalse();
  }

  /**
   * Test that the handler correctly supports http error codes i.e. 404 (NOT FOUND) with a
   * Content-Length header.
   */
  @Test
  public void httpErrorsWithContentAreSupported() {
    EmbeddedChannel ch = new EmbeddedChannel(new HttpUploadHandler(null, ImmutableList.of()));
    ByteArrayInputStream data = new ByteArrayInputStream(new byte[] {1, 2, 3, 4, 5});
    ChannelPromise writePromise = ch.newPromise();
    ch.writeOneOutbound(new UploadCommand(CACHE_URI, true, "abcdef", data, 5), writePromise);

    HttpRequest request = ch.readOutbound();
    assertThat(request).isInstanceOf(HttpRequest.class);
    HttpChunkedInput content = ch.readOutbound();
    assertThat(content).isInstanceOf(HttpChunkedInput.class);

    ByteBuf errorMsg = ByteBufUtil.writeAscii(ch.alloc(), "error message");
    FullHttpResponse response =
        new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.NOT_FOUND, errorMsg);
    response.headers().set(HttpHeaders.CONNECTION, HttpHeaderValues.KEEP_ALIVE);

    ch.writeInbound(response);

    assertThat(writePromise.isDone()).isTrue();
    assertThat(writePromise.cause()).isInstanceOf(HttpException.class);
    assertThat(((HttpException) writePromise.cause()).response().status())
        .isEqualTo(HttpResponseStatus.NOT_FOUND);
    assertThat(ch.isOpen()).isTrue();
  }
}
