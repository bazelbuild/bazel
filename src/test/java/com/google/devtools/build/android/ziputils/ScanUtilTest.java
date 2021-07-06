// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.ziputils;

import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.assertThrows;

import java.nio.ByteBuffer;
import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link ScanUtil}.
 */
@RunWith(JUnit4.class)
public class ScanUtilTest {

  @Test
  public void testScanTo() {
    assertThrows(NullPointerException.class, () -> assertLocation(null, new byte[] {}, -1));
    assertThrows(NullPointerException.class, () -> assertLocation(new byte[] {}, null, -1));
    assertLocation(new byte[] {}, new byte[] {}, -1);
    assertLocation(new byte[] {}, new byte[] {1}, 0);
    assertLocation(new byte[] {}, new byte[] {1, 2, 3, 4}, 0);
    assertLocation(new byte[] {1}, new byte[] {}, -1);
    assertLocation(new byte[] {1}, new byte[] {1}, 0);
    assertLocation(new byte[] {1}, new byte[] {1, 2, 3, 4}, 0);
    assertLocation(new byte[] {1}, new byte[] {5, 4, 1, 2, 3, 4}, 2);
    assertLocation(new byte[] {1}, new byte[] {4, 2, 3, 1}, 3);
    assertLocation(new byte[] {1, 2, 3, 4}, new byte[] {}, -1);
    assertLocation(new byte[] {1, 2, 3, 4}, new byte[] {1}, -1);
    assertLocation(new byte[] {1, 2, 3, 4}, new byte[] {1, 2, 3, 4}, 0);
    assertLocation(new byte[] {1, 2, 3, 4}, new byte[] {1, 2, 3, 4, 1, 2, 3, 4}, 0);
    assertLocation(new byte[] {1, 2, 3, 4}, new byte[] {5, 1, 2, 3, 4}, 1);
    assertLocation(new byte[] {1, 2, 3, 4}, new byte[] {5, 1, 2, 3, 4, 5}, 1);
    assertLocation(new byte[] {1, 2, 3, 4}, new byte[] {5, 5, 1, 2, 3, 4}, 2);
    assertLocation(new byte[] {1, 2, 3, 4}, new byte[] {5, 1, 1, 2, 3, 4}, 2);
    assertLocation(new byte[] {1, 2, 3, 4}, new byte[] {5, 1, 2, 3, 5, 4}, -1);
  }

  @Test
  public void testScanBackwardsTo() {
    assertThrows(
        NullPointerException.class, () -> assertBackwardsLocation(null, new byte[] {}, -1));
    assertThrows(
        NullPointerException.class, () -> assertBackwardsLocation(new byte[] {}, null, -1));
    assertBackwardsLocation(new byte[] {}, new byte[] {}, -1);
    assertBackwardsLocation(new byte[] {}, new byte[] {1}, 0);
    assertBackwardsLocation(new byte[] {}, new byte[] {1, 2, 3, 4}, 3);
    assertBackwardsLocation(new byte[] {1}, new byte[] {}, -1);
    assertBackwardsLocation(new byte[] {1}, new byte[] {1}, 0);
    assertBackwardsLocation(new byte[] {1}, new byte[] {1, 2, 3, 4}, 0);
    assertBackwardsLocation(new byte[] {1}, new byte[] {5, 4, 1, 2, 3, 4}, 2);
    assertBackwardsLocation(new byte[] {1}, new byte[] {4, 2, 3, 1}, 3);
    assertBackwardsLocation(new byte[] {1, 2, 3, 4}, new byte[] {}, -1);
    assertBackwardsLocation(new byte[] {1, 2, 3, 4}, new byte[] {1}, -1);
    assertBackwardsLocation(new byte[] {1, 2, 3, 4}, new byte[] {1, 2, 3, 4}, 0);
    assertBackwardsLocation(new byte[] {1, 2, 3, 4}, new byte[] {1, 2, 3, 4, 1, 2, 3, 4}, 4);
    assertBackwardsLocation(new byte[] {1, 2, 3, 4}, new byte[] {1, 2, 3, 4, 1, 2, 3, 4, 1}, 4);
    assertBackwardsLocation(new byte[] {1, 2, 3, 4}, new byte[] {5, 1, 2, 3, 4}, 1);
    assertBackwardsLocation(new byte[] {1, 2, 3, 4}, new byte[] {5, 1, 2, 3, 4, 5}, 1);
    assertBackwardsLocation(new byte[] {1, 2, 3, 4}, new byte[] {5, 5, 1, 2, 3, 4}, 2);
    assertBackwardsLocation(new byte[] {1, 2, 3, 4}, new byte[] {5, 1, 1, 2, 3, 4}, 2);
    assertBackwardsLocation(new byte[] {1, 2, 3, 4}, new byte[] {5, 1, 2, 3, 5, 4}, -1);
  }

  private void assertLocation(byte[] target, byte[] domain, int expected) {
    int pos = ScanUtil.scanTo(target, domain != null ? ByteBuffer.wrap(domain) : null);
    assertWithMessage("Position of " + Arrays.toString(target) + " in " + Arrays.toString(domain))
        .that(pos)
        .isEqualTo(expected);
  }

  private void assertBackwardsLocation(byte[] target, byte[] domain, int expected) {
    ByteBuffer buf = null;
    if (domain != null) {
      buf = ByteBuffer.wrap(domain);
      buf.position(buf.limit());
    }
    int pos = ScanUtil.scanBackwardsTo(target, buf);
    assertWithMessage(
            "Position of "
                + Arrays.toString(target)
                + " in "
                + Arrays.toString(domain)
                + ", "
                + buf.position()
                + ", "
                + buf.limit())
        .that(pos)
        .isEqualTo(expected);
  }
}
