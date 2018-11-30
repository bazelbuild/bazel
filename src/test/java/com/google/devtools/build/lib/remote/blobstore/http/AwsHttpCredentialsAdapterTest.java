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

package com.google.devtools.build.lib.remote.blobstore.http;

import com.amazonaws.auth.AWSCredentialsProvider;
import com.amazonaws.auth.AWSStaticCredentialsProvider;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.auth.BasicSessionCredentials;
import com.google.common.base.CharMatcher;
import com.google.common.base.Splitter;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.packages.util.ResourceLoader;
import io.netty.handler.codec.http.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.net.URI;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.function.BiConsumer;
import java.util.function.Function;

import static com.google.common.truth.Truth.assertThat;

@RunWith(JUnit4.class)
public class AwsHttpCredentialsAdapterTest {

  private static final String TEST_ACCESS_KEY = "AKIDEXAMPLE";
  private static final String TEST_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
  private static final String TEST_SESSION_TOKEN = "AQoDYXdzEPT//////////wEXAMPLEtc764bNrC" +
      "9SAPBSM22wDOk4x4HIZ8j4FZTwdQWLWsKWHGBuFqwAeMicRXmxfpSPfIeoIYRqTflfKD8YUuwthAx7mSEI/" +
      "qkPpKPi/kMcGdQrmGdeehM4IC1NtBmUpp2wUE8phUZampKsburEDy0KPkyQDYwT7WZ0wq5VSXDvp75YU9HF" +
      "vlRd8Tx6q6fE8YQcHNVXAkiY9q6d+xo0rKwT38xVqr7ZD0u0iPPkUL64lIZbqBAz+scqKmlzm8FDrypNC9Y" +
      "jc8fPOLn9FX9KSYvKTr4rvx3iSIlTJabIQwj2ICCR/oLxBA==";

  private static final String TEST_REGION = "us-east-1";
  private static final String TEST_SERVICE = "service";

  private static final String NEWLINE = "\n";
  private static final Splitter NEWLINE_SPLITTER = Splitter.on(NEWLINE).omitEmptyStrings();
  private static final Splitter REQLINE_SPLITTER = Splitter.on(CharMatcher.whitespace());
  private static final Splitter HDR_SPLITTER = Splitter.on(":");

  private AWSCredentialsProvider basicCreds = new AWSStaticCredentialsProvider(
      new BasicAWSCredentials(TEST_ACCESS_KEY, TEST_SECRET_KEY)
  );

  private AWSCredentialsProvider sessionTokenCreds = new AWSStaticCredentialsProvider(
      new BasicSessionCredentials(TEST_ACCESS_KEY, TEST_SECRET_KEY, TEST_SESSION_TOKEN)
  );

  private TestVector parseTestVector(final AWSCredentialsProvider creds, final String vectorPath) throws Exception {
    final String vectorName = vectorPath.substring(vectorPath.lastIndexOf('/') + 1);

    String testPath = "third_party/aws-sig-v4-test-suite/" + vectorPath;

    final List<String> rawRequest = NEWLINE_SPLITTER.splitToList(ResourceLoader.readFromResources(
      testPath + "/" + vectorName + ".req"));
    final Function<String, String> readTestData = (name) -> {
      try {
        final String testFile = testPath + "/" + vectorName + "." + name;
        return ResourceLoader.readFromResources(testFile);
      } catch (final IOException ioe) {
        throw new RuntimeException(ioe);
      }
    };

    final Iterator<String> reqLine = REQLINE_SPLITTER.split(rawRequest.get(0)).iterator();
    final HttpMethod verb = HttpMethod.valueOf(reqLine.next());
    final String resource = reqLine.next();
    final HttpVersion version = HttpVersion.valueOf(reqLine.next());

    final Iterator<Map.Entry<String, String>> rawHeaders =
      rawRequest.subList(1, rawRequest.size())
        .stream()
        .map(HDR_SPLITTER::splitToList)
        .map(x -> Maps.immutableEntry(x.get(0), x.get(1)))
        .iterator();

    final HttpHeaders headers = new DefaultHttpHeaders();
    final Map<String, List<String>> parsedHeaders = new HashMap<>();
    rawHeaders
        .forEachRemaining(entry -> {
          final String key = entry.getKey();
          final String value = entry.getValue();
          parsedHeaders.computeIfAbsent(key, _x -> new ArrayList<>()).add(value);
        });

    parsedHeaders.forEach((BiConsumer<String, Iterable<String>>) headers::add);

    final HttpRequest request = new DefaultHttpRequest(version, verb, resource, headers);

    final String canonicalRequest = readTestData.apply("creq");
    final String stringToSign = readTestData.apply("sts");
    final String authString = readTestData.apply("authz");

    final URI uri = URI.create(String.format("https://%s%s", headers.get("Host"), resource));

    final ZonedDateTime date = ZonedDateTime.parse(headers.get("X-Amz-Date"),
        DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss'Z'").withZone(ZoneId.of("UTC")));

    final AwsHttpCredentialsAdapter adapter =
        new AwsHttpCredentialsAdapter(TEST_REGION, null, TEST_SERVICE, creds, false,
            "my-header1", "my-header2");

    return new TestVector(creds, adapter, request, canonicalRequest, stringToSign, authString, uri, date);
  }

  @Test
  public void testPostVanillaVector() throws Exception {
    parseTestVector(basicCreds, "post-vanilla")
      .assertCanonicalRequest()
      .assertStringToSign()
      .assertAuthHeader();
  }

  @Test
  public void getHeaderKeyDuplicate() throws Exception {
    parseTestVector(basicCreds, "get-header-key-duplicate")
      .assertCanonicalRequest()
      .assertStringToSign()
      .assertAuthHeader();
  }

  @Test
  @Ignore("Right now this is dependent on the test suite to parse correctly")
  public void getHeaderValueMultiline() throws Exception {
    parseTestVector(basicCreds, "get-header-value-multiline")
     .assertCanonicalRequest()
     .assertStringToSign()
     .assertAuthHeader();
  }

  @Test
  public void getHeaderValueOrder() throws Exception {
    parseTestVector(basicCreds, "get-header-value-order")
     .assertCanonicalRequest()
     .assertStringToSign()
     .assertAuthHeader();
  }

  @Test
  public void getHeaderValueTrim() throws Exception {
    parseTestVector(basicCreds, "get-header-value-trim")
      .assertCanonicalRequest()
      .assertStringToSign()
      .assertAuthHeader();
  }

  @Test
  public void getUnreserved() throws Exception {
    parseTestVector(basicCreds, "get-unreserved")
      .assertCanonicalRequest()
      .assertStringToSign()
      .assertAuthHeader();
  }

  @Test
  public void getUtf8() throws Exception {
    parseTestVector(basicCreds, "get-utf8")
      .assertCanonicalRequest()
      .assertStringToSign()
      .assertAuthHeader();
  }

  @Test
  public void getVanilla() throws Exception {
    parseTestVector(basicCreds, "get-vanilla")
      .assertCanonicalRequest()
      .assertStringToSign()
      .assertAuthHeader();
  }

  @Test
  public void getVanillaEmptyQueryKey() throws Exception {
    parseTestVector(basicCreds, "get-vanilla-empty-query-key")
      .assertCanonicalRequest()
      .assertStringToSign()
      .assertAuthHeader();
  }

  @Test
  public void getVanillaQuery() throws Exception {
    parseTestVector(basicCreds, "get-vanilla-query")
      .assertCanonicalRequest()
      .assertStringToSign()
      .assertAuthHeader();
  }

  @Test
  public void getVanillaQueryOrderKey() throws Exception {
    parseTestVector(basicCreds, "get-vanilla-query-order-key")
      .assertCanonicalRequest()
      .assertStringToSign()
      .assertAuthHeader();
  }

  @Test
  public void getVanillaQueryOrderKeyCase() throws Exception {
    parseTestVector(basicCreds, "get-vanilla-query-order-key-case")
      .assertCanonicalRequest()
      .assertStringToSign()
      .assertAuthHeader();
  }

  @Test
  public void getVanillaQueryOrderValue() throws Exception {
    parseTestVector(basicCreds, "get-vanilla-query-order-value")
      .assertCanonicalRequest()
      .assertStringToSign()
      .assertAuthHeader();
  }

  @Test
  public void getVanillaQueryUnreserved() throws Exception {
    parseTestVector(basicCreds, "get-vanilla-query-unreserved")
      .assertCanonicalRequest()
      .assertStringToSign()
      .assertAuthHeader();
  }

  @Test
  public void getVanillaUtf8Query() throws Exception {
    parseTestVector(basicCreds, "get-vanilla-utf8-query")
      .assertCanonicalRequest()
      .assertStringToSign()
      .assertAuthHeader();
  }

  @Test
  public void postHeaderKeyCase() throws Exception {
    parseTestVector(basicCreds, "post-header-key-case")
      .assertCanonicalRequest()
      .assertStringToSign()
      .assertAuthHeader();
  }

  @Test
  public void postHeaderKeySort() throws Exception {
    parseTestVector(basicCreds, "post-header-key-sort")
      .assertCanonicalRequest()
      .assertStringToSign()
      .assertAuthHeader();
  }

  @Test
  public void postHeaderValueCase() throws Exception {
    parseTestVector(basicCreds, "post-header-value-case")
      .assertCanonicalRequest()
      .assertStringToSign()
      .assertAuthHeader();
  }

  @Test
  public void postVanilla() throws Exception {
    parseTestVector(basicCreds, "post-vanilla")
      .assertCanonicalRequest()
      .assertStringToSign()
      .assertAuthHeader();
  }

  @Test
  public void postVanillaEmptyQueryValue() throws Exception {
    parseTestVector(basicCreds, "post-vanilla-empty-query-value")
      .assertCanonicalRequest()
      .assertStringToSign()
      .assertAuthHeader();
  }

  @Test
  public void postVanillaQuery() throws Exception {
    parseTestVector(basicCreds, "post-vanilla-query")
      .assertCanonicalRequest()
      .assertStringToSign()
      .assertAuthHeader();
  }

  @Test
  public void postStsHeaderBefore() throws Exception {
    parseTestVector(basicCreds, "post-sts-token/post-sts-header-before")
        .assertCanonicalRequest()
        .assertStringToSign()
        .assertAuthHeader();
  }

  @Test
  public void postStsHeaderAfter() throws Exception {
    parseTestVector(sessionTokenCreds, "post-sts-token/post-sts-header-after")
        .assertCanonicalRequest()
        .assertStringToSign()
        .assertAuthHeader();
  }

}

class TestVector {
  final AWSCredentialsProvider creds;
  final AwsHttpCredentialsAdapter adapter;
  final HttpRequest request;
  final String expectedCanonicalRequest;
  final String expectedStringToSign;
  final String expectedAuthString;
  final URI uri;
  final String amzDate;
  final ZonedDateTime date;

  TestVector(final AWSCredentialsProvider creds, final AwsHttpCredentialsAdapter adapter,
             final HttpRequest request, final String expectedCanonicalRequest,
             final String expectedStringToSign, final String expectedAuthString, final URI uri,
             final ZonedDateTime date) {
    this.creds = creds;
    this.adapter = adapter;
    this.request = request;
    this.expectedCanonicalRequest = expectedCanonicalRequest;
    this.expectedStringToSign = expectedStringToSign;
    this.expectedAuthString = expectedAuthString;
    this.uri = uri;
    this.date = date;
    this.amzDate = this.date.format(DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss'Z'"));
  }

  public TestVector assertCanonicalRequest() throws Exception {
    final String actual = adapter.canonicalRequest(request, amzDate);
    assertThat(actual)
        .named("Canonical Request")
        .isEqualTo(expectedCanonicalRequest);
    return this;
  }

  public TestVector assertStringToSign() throws Exception {
    final String actual = adapter.stringToSign(request, date, amzDate);
    assertThat(actual)
        .named("String to sign")
        .isEqualTo(expectedStringToSign);
    return this;
  }

  public TestVector assertAuthHeader() throws Exception {
    final String actual = adapter.authorizationHeader(creds.getCredentials(), request, date, amzDate);
    assertThat(actual)
        .named("Auth Header")
        .isEqualTo(expectedAuthString);
    return this;
  }
}
