// Copyright 2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.actions.cache;


import com.google.common.io.BaseEncoding;
import com.google.common.testing.EqualsTester;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class MetadataTest {

  private static byte[] toBytes(String hex) {
    return BaseEncoding.base16().upperCase().decode(hex);
  }

  @Test
  public void testEqualsAndHashCode() throws Exception {
    // Each "equality group" is checked for equality within itself (including hashCode equality)
    // and inequality with members of other equality groups.
    new EqualsTester()
        .addEqualityGroup(new Metadata(toBytes("00112233445566778899AABBCCDDEEFF")),
                          new Metadata(toBytes("00112233445566778899AABBCCDDEEFF")))
        .addEqualityGroup(new Metadata(1))
        .addEqualityGroup(new Metadata(toBytes("FFFFFF00000000000000000000000000")))
        .addEqualityGroup(new Metadata(2),
                          new Metadata(2))
        .addEqualityGroup("a string")
        .testEquals();
  }
}
