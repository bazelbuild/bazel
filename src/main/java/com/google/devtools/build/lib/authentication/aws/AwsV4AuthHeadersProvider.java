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

import com.amazonaws.auth.AWSCredentials;
import com.amazonaws.auth.AWSCredentialsProvider;
import com.amazonaws.auth.AWSSessionCredentials;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.CharMatcher;
import com.google.common.base.Charsets;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.base.Strings;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ComparisonChain;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Maps;
import com.google.common.escape.Escaper;
import com.google.common.hash.HashCode;
import com.google.common.hash.Hashing;
import com.google.common.net.HttpHeaders;
import com.google.common.net.UrlEscapers;
import com.google.devtools.build.lib.runtime.AuthHeadersProvider;
import com.google.devtools.build.lib.runtime.AuthHeaderRequest;

import java.io.IOException;
import java.net.URI;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Optional;
import java.util.List;
import java.util.Map;
import java.util.Collection;
import java.util.HashMap;
import java.util.Arrays;
import java.util.function.BiConsumer;
import java.util.stream.Stream;

import static java.time.ZoneOffset.UTC;

/**
 * Provider for creating AWS credentials as Http Authentication headers
 *
 * This follows the specification found in
 * https://docs.aws.amazon.com/AmazonS3/latest/API/sigv4-auth-using-authorization-header.html
 */
class AwsV4AuthHeadersProvider implements AuthHeadersProvider {

  private static final String AWS_SIGNATURE_VERSION = "AWS4-HMAC-SHA256";
  private static final String AWS_HDR_PREFIX = "x-amz-";
  private static final String AWS_DATE_HDR = "x-amz-date";
  private static final String AWS_STS_HEADER = "x-amz-security-token";
  private static final String AWS_CONTENT_SHA_HDR = "x-amz-content-sha256";
  private static final String UNSIGNED_PAYLOAD_SIG = "UNSIGNED-PAYLOAD";
  private static final String BARE_PAYLOAD_HASH = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

  private static final String AWS4_REQUEST = "aws4_request";

  private static final DateTimeFormatter iso8601DateFormatter = DateTimeFormatter
      .ofPattern("yyyyMMdd'T'HHmmss'Z'")
      .withZone(UTC);
  private static final DateTimeFormatter signDateFormatter = DateTimeFormatter
      .ofPattern("yyyyMMdd")
      .withZone(UTC);

  private static final Escaper QRY_PARAM_ESCAPER = UrlEscapers.urlPathSegmentEscaper();

  private static final Splitter PARAM_SPLITTER = Splitter.on('&');
  private static final Splitter KEY_VAL_SPLITTER = Splitter.on('=').limit(1);
  private static final Splitter PATH_SPLITTER = Splitter.on('/');
  private static final Joiner PARAM_JOINER = Joiner.on('&');
  private static final Joiner KEY_VAL_JOINER = Joiner.on('=');
  private static final Joiner.MapJoiner HDR_JOINER = Joiner.on('\n').withKeyValueSeparator(':');
  private static final Joiner SIGN_HDR_JOINER = Joiner.on(';');
  private static final Joiner NEWLINE_JOINER = Joiner.on('\n');
  private static final Joiner PATH_JOINER = Joiner.on('/');
  private static final Joiner COMMA_JOINER = Joiner.on(',');
  public static final Escaper URL_PATH_SEGMENT_ESCAPER = UrlEscapers.urlPathSegmentEscaper();

  private final AWSCredentialsProvider awsCredsProvider;
  private final AwsRegion region;
  private final String service;
  private final ImmutableSet<String> headersToInclude;
  private final boolean useUnsignedPayloads;

  AwsV4AuthHeadersProvider(final AwsRegion region, final String service,
      final AWSCredentialsProvider awsCredsProvider,
      final boolean useUnsignedPayloads, final String... additionalSignHeaders) {
    this.awsCredsProvider = Preconditions.checkNotNull(awsCredsProvider);
    Preconditions.checkArgument(!Strings.isNullOrEmpty(service));
    this.service = service;
    this.useUnsignedPayloads = useUnsignedPayloads;
    this.region = Preconditions.checkNotNull(region);

    final ImmutableSet.Builder<String> canoicHeaderBuilder = ImmutableSet.builder();
    canoicHeaderBuilder
        .add(HttpHeaders.HOST.toLowerCase())
        .add(HttpHeaders.CONTENT_LENGTH.toLowerCase());

    Arrays.stream(additionalSignHeaders)
        .map(String::toLowerCase)
        .forEach(canoicHeaderBuilder::add);
    this.headersToInclude = canoicHeaderBuilder.build();
  }

  @Override
  public String getType() {
    return "aws-sigv4";
  }

  @Override
  public void refresh() throws IOException {
    try {
      awsCredsProvider.refresh();
    } catch (final Exception e) {
      throw new IOException("Unable to refresh AWS credentials", e);
    }
  }

  @Override
  public boolean isEnabled() {
    return true;
  }

  @Override
  public Map<String, List<String>> getRequestHeaders(final AuthHeaderRequest request) throws IOException {
      final ZonedDateTime now = ZonedDateTime.now(UTC);
      final String amzDate = now.format(iso8601DateFormatter);
      final AWSCredentials creds = awsCredsProvider.getCredentials();

      final Map<String, List<String>> headers = new HashMap<>();
      final BiConsumer<String, String> addHeader = (key, value) -> {
        headers.computeIfAbsent(key, k -> new ArrayList<>()).add(value);
      };

      Optional<String> sessionToken = Optional.empty();
      if (creds instanceof AWSSessionCredentials) {
        final String stsToken = ((AWSSessionCredentials) creds).getSessionToken();
        addHeader.accept(AWS_STS_HEADER, stsToken);
        sessionToken = Optional.ofNullable(stsToken);
      }

      final String authHeader = authorizationHeader(creds, request, sessionToken, now, amzDate);
      addHeader.accept(HttpHeaders.AUTHORIZATION, authHeader);
      addHeader.accept(AWS_DATE_HDR, amzDate);

      if (useUnsignedPayloads) {
        addHeader.accept(AWS_CONTENT_SHA_HDR, "UNSIGNED-PAYLOAD");
      }

      return headers;
  }

  @VisibleForTesting
  String authorizationHeader(final AWSCredentials creds, final AuthHeaderRequest request,
      final Optional<String> sessionToken, final ZonedDateTime date,
      final String amzDate) throws IOException {
    try {
      final String accessKey = creds.getAWSAccessKeyId();
      final String secretKey = creds.getAWSSecretKey();

      final String stringToSign = stringToSign(request, sessionToken, date, amzDate);
      final byte[] signingKey = signature(secretKey, date, region);
      final CharSequence signature = hex(hmacSHA256(signingKey, stringToSign));

      return String.format(
          "AWS4-HMAC-SHA256 Credential=%s/%s/%s/%s/%s, SignedHeaders=%s, Signature=%s",
          accessKey,
          date.format(signDateFormatter),
          region.name,
          this.service,
          AWS4_REQUEST,
          signedHeaders(request, sessionToken, amzDate),
          signature);
    } catch (NoSuchAlgorithmException nsae) {
      throw new IOException("Unable to correctly initialise crypto primitives", nsae);
    }
  }

  @VisibleForTesting
  String stringToSign(final AuthHeaderRequest request, final Optional<String> sessionToken,
          final ZonedDateTime date, final String amzDate) throws NoSuchAlgorithmException {
    return NEWLINE_JOINER.join(
        AWS_SIGNATURE_VERSION,
        amzDate,
        scope(date),
        hex(sha256Hash(canonicalRequest(request, sessionToken, amzDate)))
    );
  }

  @VisibleForTesting
  byte[] signature(final String secretKey, final ZonedDateTime date, final AwsRegion awsRegion)
      throws IOException {
    final String signDate = date.format(signDateFormatter);
    final byte[] dateKey = hmacSHA256("AWS4" + secretKey, signDate);
    final byte[] dateRegionKey = hmacSHA256(dateKey, awsRegion.name);
    final byte[] dateRegionServiceKey = hmacSHA256(dateRegionKey, this.service);
    final byte[] signingKey = hmacSHA256(dateRegionServiceKey, AWS4_REQUEST);
    return signingKey;
  }

  @VisibleForTesting
  String canonicalRequest(final AuthHeaderRequest request, final Optional<String> sessionToken,
          final String amzDate) {
    final URI uri = request.uri();
    return NEWLINE_JOINER.join(
        request.method().get().toUpperCase(),
        canonicalURI(uri),
        canonicalQueryString(uri),
        canonicalHeaders(request, sessionToken, amzDate) + "\n",
        signedHeaders(request, sessionToken, amzDate),
        hashedPayload()
    );
  }

  /**
   * Binds the resulting signature to a specific date, an AWS region, and a service.
   * Thus, your resulting signature will work only in the specific region and for a
   * specific service. The signature is valid for seven days after the specified date.
   * `date.Format(<YYYYMMDD>) + "/" + <region> + "/" + <service> + "/aws4_request"`
   *
   * For Amazon S3, the service string is s3. For a list of region strings, see
   * Regions and Endpoints in the AWS General Reference. The region column in this table
   * provides the list of valid region strings.
   *
   * The following scope restricts the resulting signature to the us-east-1 region and Amazon S3.
   *
   * `20130606/us-east-1/s3/aws4_request`
   *
   * Note:
   *  Scope must use the same date that you use to compute the signing key
   */
  private String scope(final ZonedDateTime date) {
    return PATH_JOINER.join(
        date.format(signDateFormatter),
        this.region.name,
        this.service,
        AWS4_REQUEST
    );
  }

  /**
   * The URI-encoded version of the absolute path component of the URI—everything starting
   * with the "/" that follows the domain name and up to the end of the string or to the
   * question mark character ('?') if you have query string parameters.
   *
   * The URI in the following example, `/examplebucket/myphoto.jpg`, is the absolute path
   * and you don't encode the "/" in the absolute path:
   *
   * `http://s3.amazonaws.com/examplebucket/myphoto.jpg`
   *
   * Note:
   * You do not normalize URI paths for requests to Amazon S3. For example, you may have
   * a bucket with an object named "my-object//example//photo.user".
   *
   * Normalizing the path changes the object name in the request to
   * "my-object/example/photo.user".
   *
   * This is an incorrect path for that object.
   */
  private CharSequence canonicalURI(final URI uri) {
    return PATH_JOINER.join(
        PATH_SPLITTER.splitToList(uri.getPath())
          .stream()
          .map(URL_PATH_SEGMENT_ESCAPER::escape)
          .iterator()
    );
  }

  /**
   * The URI-encoded query string parameters.
   * You URI-encode name and values individually.
   * You must also sort the parameters in the canonical query string alphabetically by
   * key name.
   * The sorting occurs after encoding.
   * The query string in the following URI example is `prefix=somePrefix&marker=someMarker&max-keys=20`:
   *
   * `http://s3.amazonaws.com/examplebucket?prefix=somePrefix&marker=someMarker&max-keys=20`
   *
   * The canonical query string is as follows (line breaks are added to this example for
   * readability):
   *
   * ```java
   *   UriEncode("marker")+"="+UriEncode("someMarker")+"&"+
   *   UriEncode("max-keys")+"="+UriEncode("20") + "&" +
   *   UriEncode("prefix")+"="+UriEncode("somePrefix")
   * ```
   *
   * When a request targets a subresource, the corresponding query parameter value will
   * be an empty string ("").
   *
   * For example, the following URI identifies the ACL subresource on the examplebucket
   * bucket:
   *  `http://s3.amazonaws.com/examplebucket?acl`
   *
   * The CanonicalQueryString in this case is as follows:
   *  `UriEncode("acl") + "=" + ""`
   *
   * If the URI does not include a '?', there is no query string in the request, and you
   * set the canonical query string to an empty string (""). You will still need to
   * include the "\n".
   */
  private CharSequence canonicalQueryString(final URI uri) {
    final StringBuilder builder = new StringBuilder();

    if (uri.getQuery() != null) {
      final Iterator<String> queryParams = PARAM_SPLITTER
          .trimResults()
          .splitToList(uri.getQuery())
          .stream()
          .map((String x) -> {
            if (x.contains("=")) {
              // You URI-encode name and values individually. You must also sort the
              // parameters in the canonical query string alphabetically by key name.
              // The sorting occurs after encoding. The query string in the following URI
              // example is `prefix=somePrefix&marker=someMarker&max-keys=20:`
              // `http://s3.amazonaws.com/examplebucket?prefix=somePrefix&marker=someMarker&max-keys=20`
              //
              // The canonical query string is as follows (line breaks are added to this example for
              // readability):
              // ```java
              //    UriEncode("marker")+"="+UriEncode("someMarker")+"&"+
              //    UriEncode("max-keys")+"="+UriEncode("20") + "&" +
              //    UriEncode("prefix")+"="+UriEncode("somePrefix")
              // ```
              return KEY_VAL_JOINER.join(
                  KEY_VAL_SPLITTER.splitToList(x)
                      .stream()
                      .map(QRY_PARAM_ESCAPER::escape)
                      .iterator()
              );
            } else {
              // When a request targets a subresource, the corresponding query parameter
              // value will be an empty string (""). For example, the following URI
              // identifies the ACL subresource on the examplebucket bucket
              // `http://s3.amazonaws.com/examplebucket?acl`
              // The CanonicalQueryString in this case is as follows:
              // `UriEncode("acl") + "=" + ""`
              return KEY_VAL_JOINER.join(QRY_PARAM_ESCAPER.escape(x), "");
            }
          })
          .sorted()
          .iterator();

      PARAM_JOINER.appendTo(builder, queryParams);
    } else {
      // If the URI does not include a '?', there is no query string in the request, and
      // you set the canonical query string to an empty string (""). You will still need
      // to include the newline
      builder.append("");
    }

    return builder.toString();
  }

  /**
   * List of request headers with their values.
   * Individual header name and value pairs are separated by the newline character ("\n").
   * Header names must be in lowercase. You must sort the header names alphabetically to
   * construct the string, as shown in the following example:
   *
   * ```java
   *   Lowercase(<HeaderName1>)+":"+Trim(<value>)+"\n"
   *   Lowercase(<HeaderName2>)+":"+Trim(<value>)+"\n"
   *   ...
   *   Lowercase(<HeaderNameN>)+":"+Trim(<value>)+"\n"
   * ```
   *
   * The CanonicalHeaders list must include the following:
   * - HTTP host header.
   * - If the Content-Type header is present in the request, you must add it to the
   *   CanonicalHeaders list.
   * - Any x-amz-* headers that you plan to include in your request must also be added.
   *   For example, if you are using temporary security credentials, you need to include
   *   x-amz-security-token in your request.
   *   You must add this header in the list of CanonicalHeaders.
   *
   * Note:
   * The x-amz-content-sha256 header is required for all AWS Signature Version 4 requests. It provides a hash of the request payload. If there is no payload, you must provide the hash of an empty string.
   *
   * The following is an example CanonicalHeaders string. The header names are in
   * lowercase and sorted.
   *
   * ```
   * host:s3.amazonaws.com
   * x-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
   * x-amz-date:20130708T220855Z
   * ```
   *
   * Note:
   * For the purpose of calculating an authorization signature, only the host and any
   * x-amz-* headers are required; however, in order to prevent data tampering,
   * you should consider including all the headers in the signature calculation.
   */
  private CharSequence canonicalHeaders(final AuthHeaderRequest request,
          final Optional<String> sessionToken, final String amzDate) {
    return HDR_JOINER.join(processHeaders(request, sessionToken, amzDate).iterator());
  }

  /**
   * An alphabetically sorted, semicolon-separated list of lowercase request header names.
   * The request headers in the list are the same headers that you included in the
   * CanonicalHeaders string. For example, for the previous example, the value of
   * SignedHeaders would be as follows:
   *
   * `host;x-amz-content-sha256;x-amz-date`
   */
  private CharSequence signedHeaders(final AuthHeaderRequest request,
          final Optional<String> sessionToken, final String amzDate) {
    final Stream<String> names = processHeaders(request, sessionToken, amzDate)
        .map(Map.Entry::getKey);
    return SIGN_HDR_JOINER.join(names.iterator());
  }

  /**
   * The hexadecimal value of the SHA256 hash of the request payload.
   * `Hex(SHA256Hash(<payload>)`
   *
   * If there is no payload in the request, you compute a hash of the empty string as follows:
   * `Hex(SHA256Hash(""))`
   *
   * The hash returns the following value:
   * `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
   *
   * For example, when you upload an object by using a PUT request, you provide object
   * data in the body.
   *
   * When you retrieve an object by using a GET request, you compute the empty string hash.
   */
  private CharSequence hashedPayload() {
    // We do not sign the payload even if we make put requests
    return useUnsignedPayloads ? UNSIGNED_PAYLOAD_SIG : BARE_PAYLOAD_HASH;
  }

  // Utility methods for supporting AWS signatures
  //------------------------------------------------------------------------------------------
  private Stream<Map.Entry<String, String>> processHeaders(final AuthHeaderRequest request,
      final Optional<String> sessionToken, final String amzDate) {
    final ListMultimap<String, String> headers = ArrayListMultimap.create(request.headers().get());
    Preconditions.checkState(!headers.get("host").isEmpty());

    if (useUnsignedPayloads && !headers.containsKey(AWS_CONTENT_SHA_HDR)) {
      headers.put(AWS_CONTENT_SHA_HDR, UNSIGNED_PAYLOAD_SIG);
    }
    if (!headers.containsKey(AWS_DATE_HDR)) {
      headers.put(AWS_DATE_HDR, amzDate);
    }
    sessionToken.ifPresent(token -> headers.put(AWS_STS_HEADER, token));

    return headers
        .asMap()
        .entrySet()
        .stream()
        .filter(x -> x.getKey() != null)
        .filter(x -> x.getKey().contains(AWS_HDR_PREFIX) || headersToInclude.contains(x.getKey()))
        .map(this::processHeader)
        .sorted((x, y) ->
          ComparisonChain.start()
              .compare(x.getKey(), y.getKey())
              .compare(x.getValue(), y.getValue())
            .result()
        );
  }

  private Map.Entry<String, String> processHeader(final Map.Entry<String, Collection<String>> header) {
    final String headerKey = header.getKey().toLowerCase();
    final String headerValue = processHeaderValues(header.getValue());
    return Maps.immutableEntry(headerKey, headerValue);
  }

  private String processHeaderValues(final Collection<String> values) {
    return COMMA_JOINER.join(values.stream().map(this::trimSpaces).iterator());
  }

  private String trimSpaces(final String value) {
    return CharMatcher.whitespace().trimAndCollapseFrom(value, ' ');
  }

  private byte[] sha256Hash(final String canonicalRequest) throws NoSuchAlgorithmException {
    final MessageDigest digest = MessageDigest.getInstance("SHA-256");
    return digest.digest(canonicalRequest.getBytes(Charsets.UTF_8));
  }

  private CharSequence hex(final byte[] bytes) {
    return HashCode.fromBytes(bytes).toString();
  }

  byte[] hmacSHA256(final String key, final String toSign) throws IOException {
    return hmacSHA256(key.getBytes(Charsets.UTF_8), toSign.getBytes(Charsets.UTF_8));
  }

  byte[] hmacSHA256(final byte[] key, final String toSign) throws IOException {
    return hmacSHA256(key, toSign.getBytes(Charsets.UTF_8));
  }

  byte[] hmacSHA256(final byte[] key, final byte[] toSign) throws IOException {
    return Hashing.hmacSha256(key).hashBytes(toSign).asBytes();
  }
}


