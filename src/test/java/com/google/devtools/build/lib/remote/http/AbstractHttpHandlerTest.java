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

import build.bazel.remote.execution.v2.Digest;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import io.netty.channel.ChannelPromise;
import io.netty.channel.embedded.EmbeddedChannel;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpRequest;
import java.io.ByteArrayOutputStream;
import java.net.URI;
import java.util.Arrays;
import java.util.Map.Entry;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link AbstractHttpHandlerTest}. */
@RunWith(JUnit4.class)
public abstract class AbstractHttpHandlerTest {

  private static final DigestUtil DIGEST_UTIL = new DigestUtil(DigestHashFunction.SHA256);
  private static final Digest DIGEST = DIGEST_UTIL.computeAsUtf8("foo");

  @Test
  public void basicAuthShouldWork() throws Exception {
    URI uri = new URI("http://user:password@does.not.exist/foo");
    EmbeddedChannel ch = new EmbeddedChannel(new HttpDownloadHandler(null, ImmutableList.of()));
    DownloadCommand cmd = new DownloadCommand(uri, true, DIGEST, new ByteArrayOutputStream());
    ChannelPromise writePromise = ch.newPromise();
    ch.writeOneOutbound(cmd, writePromise);

    HttpRequest request = ch.readOutbound();
    assertThat(request.headers().get(HttpHeaderNames.AUTHORIZATION))
        .isEqualTo("Basic dXNlcjpwYXNzd29yZA==");
  }

  @Test
  public void basicAuthShouldNotEnabled() throws Exception {
    URI uri = new URI("http://does.not.exist/foo");
    EmbeddedChannel ch = new EmbeddedChannel(new HttpDownloadHandler(null, ImmutableList.of()));
    DownloadCommand cmd = new DownloadCommand(uri, true, DIGEST, new ByteArrayOutputStream());
    ChannelPromise writePromise = ch.newPromise();
    ch.writeOneOutbound(cmd, writePromise);

    HttpRequest request = ch.readOutbound();
    assertThat(request.headers().contains(HttpHeaderNames.AUTHORIZATION)).isFalse();
  }

  @Test
  public void hostDoesntIncludePortHttp() throws Exception {
    URI uri = new URI("http://does.not.exist/foo");
    EmbeddedChannel ch = new EmbeddedChannel(new HttpDownloadHandler(null, ImmutableList.of()));
    DownloadCommand cmd = new DownloadCommand(uri, true, DIGEST, new ByteArrayOutputStream());
    ChannelPromise writePromise = ch.newPromise();
    ch.writeOneOutbound(cmd, writePromise);

    HttpRequest request = ch.readOutbound();
    assertThat(request.headers().get(HttpHeaderNames.HOST)).isEqualTo("does.not.exist");
  }

  @Test
  public void hostDoesntIncludePortHttps() throws Exception {
    URI uri = new URI("https://does.not.exist/foo");
    EmbeddedChannel ch = new EmbeddedChannel(new HttpDownloadHandler(null, ImmutableList.of()));
    DownloadCommand cmd = new DownloadCommand(uri, true, DIGEST, new ByteArrayOutputStream());
    ChannelPromise writePromise = ch.newPromise();
    ch.writeOneOutbound(cmd, writePromise);

    HttpRequest request = ch.readOutbound();
    assertThat(request.headers().get(HttpHeaderNames.HOST)).isEqualTo("does.not.exist");
  }

  @Test
  public void hostDoesIncludePort() throws Exception {
    URI uri = new URI("http://does.not.exist:8080/foo");
    EmbeddedChannel ch = new EmbeddedChannel(new HttpDownloadHandler(null, ImmutableList.of()));
    DownloadCommand cmd = new DownloadCommand(uri, true, DIGEST, new ByteArrayOutputStream());
    ChannelPromise writePromise = ch.newPromise();
    ch.writeOneOutbound(cmd, writePromise);

    HttpRequest request = ch.readOutbound();
    assertThat(request.headers().get(HttpHeaderNames.HOST)).isEqualTo("does.not.exist:8080");
  }

  @Test
  public void headersDoIncludeUserAgent() throws Exception {
    URI uri = new URI("http://does.not.exist:8080/foo");
    EmbeddedChannel ch =
        new EmbeddedChannel(new HttpDownloadHandler(/* credentials= */ null, ImmutableList.of()));
    DownloadCommand cmd =
        new DownloadCommand(uri, /* casDownload= */ true, DIGEST, new ByteArrayOutputStream());
    ChannelPromise writePromise = ch.newPromise();
    ch.writeOneOutbound(cmd, writePromise);

    HttpRequest request = ch.readOutbound();
    assertThat(request.headers().get(HttpHeaderNames.USER_AGENT)).isEqualTo("bazel/");
  }

  @Test
  public void extraHeadersAreIncluded() throws Exception {
    URI uri = new URI("http://does.not.exist:8080/foo");
    ImmutableList<Entry<String, String>> remoteHeaders =
        ImmutableList.of(
            Maps.immutableEntry("key1", "value1"), Maps.immutableEntry("key2", "value2"));

    EmbeddedChannel ch =
        new EmbeddedChannel(new HttpDownloadHandler(/* credentials= */ null, remoteHeaders));
    DownloadCommand cmd =
        new DownloadCommand(uri, /* casDownload= */ true, DIGEST, new ByteArrayOutputStream());
    ChannelPromise writePromise = ch.newPromise();
    ch.writeOneOutbound(cmd, writePromise);

    HttpRequest request = ch.readOutbound();
    assertThat(request.headers().get("key1")).isEqualTo("value1");
    assertThat(request.headers().get("key2")).isEqualTo("value2");
  }

  @Test
  public void multipleExtraHeadersAreSupported() throws Exception {
    URI uri = new URI("http://does.not.exist:8080/foo");
    ImmutableList<Entry<String, String>> remoteHeaders =
        ImmutableList.of(
            Maps.immutableEntry("key", "value1"), Maps.immutableEntry("key", "value2"));

    EmbeddedChannel ch =
        new EmbeddedChannel(new HttpDownloadHandler(/* credentials= */ null, remoteHeaders));
    DownloadCommand cmd =
        new DownloadCommand(uri, /* casDownload= */ true, DIGEST, new ByteArrayOutputStream());
    ChannelPromise writePromise = ch.newPromise();
    ch.writeOneOutbound(cmd, writePromise);

    HttpRequest request = ch.readOutbound();
    assertThat(request.headers().getAll("key")).isEqualTo(Arrays.asList("value1", "value2"));
  }
}
