// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.authandtls.credentialhelper;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Policy.CacheEntry;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.common.options.OptionsParsingResult;
import java.net.URI;
import java.time.Duration;
import java.time.Instant;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CredentialModule}. */
@RunWith(JUnit4.class)
public final class CredentialModuleTest {

  private static final URI TEST_URI = URI.create("https://example.com");
  private static final GetCredentialsResponse DEFAULT_RESPONSE =
      GetCredentialsResponse.newBuilder().build();

  private final ManualClock clock = new ManualClock();
  private final CredentialModule module = new CredentialModule(clock);
  private final Cache<URI, GetCredentialsResponse> cache = module.getCredentialCache();

  @Test
  public void putWithExplicitExpiration() {
    initModule("build", Duration.ofMinutes(60));

    var expiry = clock.now().plus(Duration.ofMinutes(123));

    cache.put(TEST_URI, GetCredentialsResponse.newBuilder().setExpires(expiry).build());
    assertInCache(TEST_URI, expiry);

    clock.advance(Duration.ofMinutes(122));
    assertInCache(TEST_URI, expiry);

    clock.advance(Duration.ofMinutes(2));
    assertNotInCache(TEST_URI);
  }

  @Test
  public void putWithNonzeroDefaultExpiration() {
    initModule("build", Duration.ofMinutes(60));

    var expiry = clock.now().plus(Duration.ofMinutes(60));

    cache.put(TEST_URI, DEFAULT_RESPONSE);
    assertInCache(TEST_URI, expiry);

    clock.advance(Duration.ofMinutes(59));
    assertInCache(TEST_URI, expiry);

    clock.advance(Duration.ofMinutes(2));
    assertNotInCache(TEST_URI);
  }

  @Test
  public void putWithZeroDefaultExpiration() {
    initModule("build", Duration.ZERO);

    cache.put(TEST_URI, DEFAULT_RESPONSE);

    assertNotInCache(TEST_URI);
  }

  @Test
  public void keepingDefaultDoesNotClearCache() {
    initModule("build", Duration.ofMinutes(60));

    cache.put(TEST_URI, DEFAULT_RESPONSE);
    assertInCache(TEST_URI, clock.now().plus(Duration.ofMinutes(60)));

    initModule("build", Duration.ofMinutes(60));

    assertInCache(TEST_URI, clock.now().plus(Duration.ofMinutes(60)));
  }

  @Test
  public void changingDefaultToSmallerValueClearsCache() {
    initModule("build", Duration.ofMinutes(60));

    cache.put(TEST_URI, DEFAULT_RESPONSE);
    assertInCache(TEST_URI, clock.now().plus(Duration.ofMinutes(60)));

    initModule("build", Duration.ofMinutes(30));

    assertNotInCache(TEST_URI);
  }

  @Test
  public void changingDefaultToLargerValueClearsCache() {
    initModule("build", Duration.ofMinutes(30));

    cache.put(TEST_URI, DEFAULT_RESPONSE);
    assertInCache(TEST_URI, clock.now().plus(Duration.ofMinutes(30)));

    initModule("build", Duration.ofMinutes(60));

    assertNotInCache(TEST_URI);
  }

  @Test
  public void cleanCommandClearsCache() {
    initModule("build", Duration.ofMinutes(60));

    cache.put(TEST_URI, DEFAULT_RESPONSE);
    assertInCache(TEST_URI, clock.now().plus(Duration.ofMinutes(60)));

    initModule("clean", Duration.ofMinutes(60));

    assertNotInCache(TEST_URI);
  }

  private void initModule(String commandName, Duration cacheTimeout) {
    module.beforeCommand(createCommandEnvironment(commandName, cacheTimeout));
  }

  private static CommandEnvironment createCommandEnvironment(
      String commandName, Duration cacheTimeout) {
    AuthAndTLSOptions authAndTlsOptions = new AuthAndTLSOptions();
    authAndTlsOptions.credentialHelperCacheTimeout = cacheTimeout;

    OptionsParsingResult optionsParsingResult = mock(OptionsParsingResult.class);
    when(optionsParsingResult.getOptions(AuthAndTLSOptions.class)).thenReturn(authAndTlsOptions);

    CommandEnvironment env = mock(CommandEnvironment.class);
    when(env.getCommandName()).thenReturn(commandName);
    when(env.getOptions()).thenReturn(optionsParsingResult);

    return env;
  }

  private void assertInCache(URI uri, Instant expiry) {
    CacheEntry<URI, GetCredentialsResponse> entry = cache.policy().getEntryIfPresentQuietly(uri);
    assertThat(entry).isNotNull();
    assertThat(fromEpochNano(entry.expiresAt())).isEqualTo(expiry);
  }

  private void assertNotInCache(URI uri) {
    assertThat(cache.policy().getEntryIfPresentQuietly(uri)).isNull();
  }

  private static Instant fromEpochNano(long nano) {
    return Instant.ofEpochSecond(0, nano);
  }
}
