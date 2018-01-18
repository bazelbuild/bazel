package com.google.devtools.build.lib.remote.blobstore.http;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.net.HttpHeaders;
import io.netty.buffer.ByteBufAllocator;
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

@RunWith(JUnit4.class)
public class HttpUploadHandlerTest {

  private static final URI CACHE_URI = URI.create("http://storage.googleapis.com:80/cache-bucket");

  @Test
  public void uploadsShouldWork() throws Exception {

    // Test that uploading blobs works to both the Action Cache and the CAS. Also test
    // that the handler is reusable.

    EmbeddedChannel ch = new EmbeddedChannel(new HttpUploadHandler(null));
    uploadsShouldWork(true, ch);
    uploadsShouldWork(false, ch);
  }

  private void uploadsShouldWork(boolean casUpload, EmbeddedChannel ch) throws Exception {
    ByteArrayInputStream data = new ByteArrayInputStream(new byte[] {1, 2, 3, 4, 5});
    ChannelPromise writePromise = ch.newPromise();
    ch.writeOneOutbound(new UploadCommand(CACHE_URI, casUpload, "abcdef", data, 5), writePromise);

    HttpRequest request = ch.readOutbound();
    assertThat(request.method()).isEqualTo(HttpMethod.PUT);
    assertThat(request.headers().get(HttpHeaders.CONNECTION))
        .isEqualTo(HttpHeaderValues.KEEP_ALIVE.toString());

    HttpChunkedInput content = ch.readOutbound();
    assertThat(content.readChunk(ByteBufAllocator.DEFAULT).content().readableBytes()).isEqualTo(5);

    FullHttpResponse response =
        new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.OK);
    response.headers().set(HttpHeaderNames.CONNECTION, HttpHeaderValues.KEEP_ALIVE);

    ch.writeInbound(response);

    assertThat(writePromise.isDone()).isTrue();
    assertThat(ch.isOpen()).isTrue();
  }

  @Test
  public void httpErrorsAreSupported() throws Exception {

    // Test that the handler correctly supports http error codes i.e. 404 (NOT FOUND).

    EmbeddedChannel ch = new EmbeddedChannel(new HttpUploadHandler(null));
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
    assertThat(((HttpException) writePromise.cause()).status())
        .isEqualTo(HttpResponseStatus.FORBIDDEN);
    assertThat(ch.isOpen()).isFalse();
  }
}
