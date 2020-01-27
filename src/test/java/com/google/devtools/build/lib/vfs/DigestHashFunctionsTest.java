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
package com.google.devtools.build.lib.vfs;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameter;
import org.junit.runners.Parameterized.Parameters;

/**
 * Tests different {@link DigestHashFunction} for consistency between the MessageDigests and the
 * HashFunctions that it exposes.
 */
@RunWith(Parameterized.class)
public class DigestHashFunctionsTest {
  @Parameters(name = "{index}: digestHashFunction={0}")
  public static Collection<DigestHashFunction[]> hashFunctions() {
    // TODO(b/112537387): Remove the array-ification and return Collection<DigestHashFunction>. This
    // is possible in Junit4.12, but 4.11 requires the array. Bazel 0.18 will have Junit4.12, so
    // this can change then.
    return DigestHashFunction.getPossibleHashFunctions()
        .stream()
        .map(dhf -> new DigestHashFunction[] {dhf})
        .collect(ImmutableList.toImmutableList());
  }

  @Parameter public DigestHashFunction digestHashFunction;

  private void assertHashFunctionAndMessageDigestEquivalentForInput(byte[] input) {
    byte[] hashFunctionOutput = digestHashFunction.getHashFunction().hashBytes(input).asBytes();
    byte[] messageDigestOutput = digestHashFunction.cloneOrCreateMessageDigest().digest(input);
    assertThat(hashFunctionOutput).isEqualTo(messageDigestOutput);
  }

  @Test
  public void emptyDigestIsConsistent() {
    assertHashFunctionAndMessageDigestEquivalentForInput(new byte[] {});
  }

  @Test
  public void shortDigestIsConsistent() {
    assertHashFunctionAndMessageDigestEquivalentForInput("Bazel".getBytes(StandardCharsets.UTF_8));
  }
}
