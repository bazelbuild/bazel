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

package com.google.devtools.build.remote.worker.http;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link
 * com.google.devtools.build.remote.worker.http.AbstractHttpCacheServerHandler}
 */
@RunWith(JUnit4.class)
public class AbstractHttpCacheServerHandlerTest {

  @Test
  public void testValidUri() {
    assertThat(
            AbstractHttpCacheServerHandler.isUriValid(
                "http://some-path.co.uk:8080/ac/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"))
        .isTrue();
    assertThat(
            AbstractHttpCacheServerHandler.isUriValid(
                "http://127.12.12.0:8080/ac/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"))
        .isTrue();
    assertThat(
            AbstractHttpCacheServerHandler.isUriValid(
                "http://localhost:8080/ac/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"))
        .isTrue();
    assertThat(
            AbstractHttpCacheServerHandler.isUriValid(
                "https://localhost:8080/ac/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"))
        .isTrue();
    assertThat(
            AbstractHttpCacheServerHandler.isUriValid(
                "localhost:8080/ac/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"))
        .isTrue();
    assertThat(
            AbstractHttpCacheServerHandler.isUriValid(
                "ac/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"))
        .isTrue();

    assertThat(
            AbstractHttpCacheServerHandler.isUriValid(
                "http://some-path.co.uk:8080/cas/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"))
        .isTrue();
    assertThat(
            AbstractHttpCacheServerHandler.isUriValid(
                "http://127.12.12.0:8080/cas/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"))
        .isTrue();
    assertThat(
            AbstractHttpCacheServerHandler.isUriValid(
                "http://localhost:8080/cas/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"))
        .isTrue();
    assertThat(
            AbstractHttpCacheServerHandler.isUriValid(
                "https://localhost:8080/cas/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"))
        .isTrue();
    assertThat(
            AbstractHttpCacheServerHandler.isUriValid(
                "localhost:8080/cas/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"))
        .isTrue();
    assertThat(
            AbstractHttpCacheServerHandler.isUriValid(
                "cas/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"))
        .isTrue();
  }

  @Test
  public void testInvalidUri() {
    assertThat(
            AbstractHttpCacheServerHandler.isUriValid(
                "http://localhost:8080/ac_e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"))
        .isFalse();
    assertThat(
            AbstractHttpCacheServerHandler.isUriValid(
                "http://localhost:8080/cas_e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"))
        .isFalse();
    assertThat(
            AbstractHttpCacheServerHandler.isUriValid(
                "http://localhost:8080/ac/111111111111111111111"))
        .isFalse();
    assertThat(
            AbstractHttpCacheServerHandler.isUriValid(
                "http://localhost:8080/cas/111111111111111111111"))
        .isFalse();
    assertThat(
            AbstractHttpCacheServerHandler.isUriValid("http://localhost:8080/cas/823rhf&*%OL%_^"))
        .isFalse();
    assertThat(AbstractHttpCacheServerHandler.isUriValid("http://localhost:8080/ac/823rhf&*%OL%_^"))
        .isFalse();
  }
}
