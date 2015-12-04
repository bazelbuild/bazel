// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import static org.junit.Assert.assertEquals;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Map;

/** Unit tests for {@link UserUtils}. */
@RunWith(JUnit4.class)
@TestSpec(size = Suite.SMALL_TESTS)
public class UserUtilsTest {

  private static final String USER = "real_user";
  private static final Map<String, String> CLIENT_ENV = ImmutableMap
      .of("BLAZE_ORIGINATING_USER", USER);
  public static final ImmutableMap<String, String> EMPTY_CLIENT_ENV = ImmutableMap.of();


  @Test
  public void testGetOriginatingUserFromFlag() throws Exception {
    assertEquals(USER, UserUtils.getOriginatingUser(USER, EMPTY_CLIENT_ENV));
  }

  @Test
  public void testGetOriginatingUserFromClientEnv() throws Exception {
    assertEquals(USER, UserUtils.getOriginatingUser("", CLIENT_ENV));
  }

  @Test
  public void testGetOriginatingUserFromUserUtils() {
    assertEquals(UserUtils.getUserName(), UserUtils.getOriginatingUser("", EMPTY_CLIENT_ENV));
  }
}
