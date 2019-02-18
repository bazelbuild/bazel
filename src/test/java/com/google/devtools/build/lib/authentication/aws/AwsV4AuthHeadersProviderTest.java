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

package com.google.devtools.build.lib.authentication.aws;

import com.amazonaws.auth.*;
import com.google.common.base.CharMatcher;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMultimap;
import com.google.devtools.build.lib.packages.util.ResourceLoader;
import com.google.devtools.build.lib.runtime.AuthHeaderRequest;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.net.URI;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.function.Function;

import static com.google.common.truth.Truth.assertThat;

@RunWith(JUnit4.class)
public class AwsV4AuthHeadersProviderTest {

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

    final String resources = ResourceLoader.readFromResources(testPath + "/" + vectorName + ".req");
    final List<String> rawRequest = NEWLINE_SPLITTER.splitToList(resources);
    final Function<String, String> readTestData = (name) -> {
      try {
        final String testFile = testPath + "/" + vectorName + "." + name;
        return ResourceLoader.readFromResources(testFile);
      } catch (final IOException ioe) {
        throw new RuntimeException(ioe);
      }
    };

    final Iterator<String> reqLine = REQLINE_SPLITTER.split(rawRequest.get(0)).iterator();
    final String verb = reqLine.next();
    final String resource = reqLine.next();
    @SuppressWarnings("unused") final String _version = reqLine.next();

    final ImmutableListMultimap<String, String> headers =
        rawRequest.subList(1, rawRequest.size())
          .stream()
          .map(HDR_SPLITTER::splitToList)
          .collect(ImmutableListMultimap.toImmutableListMultimap(x -> x.get(0).toLowerCase(), x -> x.get(1)));

    final String host = headers.get("host").get(0);
    final String rawUri = String.format("https://%s%s", host, resource);
    final URI uri = URI.create(rawUri);

    final AuthHeaderRequest request = new AuthHeaderRequest() {
      @Override
      public boolean isHttp() {
        return true;
      }

      @Override
      public Optional<String> method() {
        return Optional.of(verb);
      }

      @Override
      public Optional<ImmutableMultimap<String, String>> headers() {
        return Optional.of(headers);
      }

      @Override
      public URI uri() {
        return uri;
      }
    };

    final String canonicalRequest = readTestData.apply("creq");
    final String stringToSign = readTestData.apply("sts");
    final String authString = readTestData.apply("authz");

    final ZonedDateTime date = ZonedDateTime.parse(headers.get("x-amz-date").get(0),
        DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss'Z'").withZone(ZoneId.of("UTC")));

    final AwsV4AuthHeadersProvider adapter =
        new AwsV4AuthHeadersProvider(AwsRegion.fromName(TEST_REGION), TEST_SERVICE, creds, false,
            "my-header1", "my-header2");

    final Optional<String> stsToken = Optional.empty();

    return new TestVector(creds, adapter, request, stsToken, canonicalRequest, stringToSign,
        authString, date);
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
  private final AWSCredentialsProvider creds;
  private final AwsV4AuthHeadersProvider provider;
  private final AuthHeaderRequest request;
  private final Optional<String> stsToken;
  private final String expectedCanonicalRequest;
  private final String expectedStringToSign;
  private final String expectedAuthString;
  private final String amzDate;
  private final ZonedDateTime date;

  TestVector(final AWSCredentialsProvider creds, final AwsV4AuthHeadersProvider provider,
             final AuthHeaderRequest request, final Optional<String> stsToken,
             final String expectedCanonicalRequest, final String expectedStringToSign,
             final String expectedAuthString, final ZonedDateTime date) {
    this.creds = creds;
    this.provider = provider;
    this.request = request;
    this.stsToken = stsToken;
    this.expectedCanonicalRequest = expectedCanonicalRequest;
    this.expectedStringToSign = expectedStringToSign;
    this.expectedAuthString = expectedAuthString;
    this.date = date;
    this.amzDate = this.date.format(DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss'Z'"));
  }

  TestVector assertCanonicalRequest() throws Exception {
    final String actual = provider.canonicalRequest(request, stsToken, amzDate);
    assertThat(actual)
        .named("Canonical Request")
        .isEqualTo(expectedCanonicalRequest);
    return this;
  }

  TestVector assertStringToSign() throws Exception {
    final String actual = provider.stringToSign(request, stsToken, date, amzDate);
    assertThat(actual)
        .named("String to sign")
        .isEqualTo(expectedStringToSign);
    return this;
  }

  TestVector assertAuthHeader() throws Exception {
    final String actual = provider
        .authorizationHeader(creds.getCredentials(), request, stsToken, date, amzDate);
    assertThat(actual)
        .named("Auth Header")
        .isEqualTo(expectedAuthString);
    return this;
  }
}
