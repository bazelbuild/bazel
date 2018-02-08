package com.google.devtools.build.lib.remote.blobstore.http;

import static com.google.common.truth.Truth.assertThat;

import io.netty.channel.ChannelPromise;
import io.netty.channel.embedded.EmbeddedChannel;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpRequest;
import java.io.ByteArrayOutputStream;
import java.net.URI;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Tests for {@link AbstractHttpHandlerTest}. */
@RunWith(JUnit4.class)
public abstract class AbstractHttpHandlerTest {

  @Test
  public void basicAuthShouldWork() throws Exception {
    URI uri = new URI("http://user:password@does.not.exist/foo");
    EmbeddedChannel ch = new EmbeddedChannel(new HttpDownloadHandler(null));
    ByteArrayOutputStream out = Mockito.spy(new ByteArrayOutputStream());
    DownloadCommand cmd = new DownloadCommand(uri, true, "abcdef",
        new ByteArrayOutputStream());
    ChannelPromise writePromise = ch.newPromise();
    ch.writeOneOutbound(cmd, writePromise);

    HttpRequest request = ch.readOutbound();
    assertThat(request.headers().get(HttpHeaderNames.AUTHORIZATION))
        .isEqualTo("Basic dXNlcjpwYXNzd29yZA==");
  }

  @Test
  public void basicAuthShouldNotEnabled() throws Exception {
    URI uri = new URI("http://does.not.exist/foo");
    EmbeddedChannel ch = new EmbeddedChannel(new HttpDownloadHandler(null));
    ByteArrayOutputStream out = Mockito.spy(new ByteArrayOutputStream());
    DownloadCommand cmd = new DownloadCommand(uri, true, "abcdef",
        new ByteArrayOutputStream());
    ChannelPromise writePromise = ch.newPromise();
    ch.writeOneOutbound(cmd, writePromise);

    HttpRequest request = ch.readOutbound();
    assertThat(request.headers().contains(HttpHeaderNames.AUTHORIZATION)).isFalse();
  }
}
