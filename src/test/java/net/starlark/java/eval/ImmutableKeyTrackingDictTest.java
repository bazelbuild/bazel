// Copyright 2021 The Bazel Authors. All rights reserved.
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

package net.starlark.java.eval;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import java.util.Map;
import net.starlark.java.eval.Dict.ImmutableKeyTrackingDict;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ImmutableKeyTrackingDict}. */
@RunWith(JUnit4.class)
public final class ImmutableKeyTrackingDictTest {

  private final ImmutableKeyTrackingDict<String, StarlarkInt> dict =
      Dict.<String, StarlarkInt>builder()
          .put("a", StarlarkInt.of(1))
          .put("b", StarlarkInt.of(2))
          .put("c", StarlarkInt.of(3))
          .put("d", StarlarkInt.of(4))
          .buildImmutableWithKeyTracking();

  @Test
  public void isImmutable() throws Exception {
    assertThat(dict.mutability()).isEqualTo(Mutability.IMMUTABLE);
    assertThrows(EvalException.class, () -> dict.putEntry("e", StarlarkInt.of(5)));
    assertThat(dict.containsKey("e")).isFalse();
    assertThat(dict.getAccessedKeys()).isEmpty();
  }

  @Test
  public void containsKey_tracksPresentKeys() {
    assertThat(dict.containsKey("a")).isTrue();
    assertThat(dict.containsKey("b")).isTrue();
    assertThat(dict.getAccessedKeys()).containsExactly("a", "b");
  }

  @Test
  public void containsKey_ignoresAbsentKeys() {
    assertThat(dict.containsKey("absent")).isFalse();
    assertThat(dict.containsKey(new Object())).isFalse();
    assertThat(dict.getAccessedKeys()).isEmpty();
  }

  @Test
  public void get_tracksPresentKeys() {
    assertThat(dict.get("a")).isEqualTo(StarlarkInt.of(1));
    assertThat(dict.get("b")).isEqualTo(StarlarkInt.of(2));
    assertThat(dict.getAccessedKeys()).containsExactly("a", "b");
  }

  @Test
  public void get_ignoresAbsentKeys() {
    assertThat(dict.get("absent")).isNull();
    assertThat(dict.get(new Object())).isNull();
    assertThat(dict.getAccessedKeys()).isEmpty();
  }

  @Test
  public void keySet_reportsAllKeys() {
    assertThat(dict.keySet()).containsExactly("a", "b", "c", "d").inOrder();
    assertThat(dict.getAccessedKeys()).isEqualTo(dict.keySet());
  }

  @Test
  public void entrySet_reportsAllKeys() {
    assertThat(dict.entrySet()).hasSize(4);
    assertThat(dict.getAccessedKeys()).isEqualTo(dict.keySet());
  }

  @Test
  public void iteration_reportsAllKeys() {
    for (String key : dict) {
      assertThat(key).isAnyOf("a", "b", "c", "d");
    }
    assertThat(dict.getAccessedKeys()).isEqualTo(dict.keySet());
  }

  @Test
  public void repr_reportsAllKeys() {
    StringBuilder sb = new StringBuilder();
    dict.repr(new Printer(sb));
    assertThat(sb.toString()).isEqualTo("{\"a\": 1, \"b\": 2, \"c\": 3, \"d\": 4}");
    assertThat(dict.getAccessedKeys()).isEqualTo(dict.keySet());
  }

  @Test
  public void mutableCopy_reportsAllKeys() {
    Map<String, StarlarkInt> copy = Dict.copyOf(Mutability.create("mutable"), dict);
    assertThat(copy).isNotSameInstanceAs(dict);
    assertThat(dict.getAccessedKeys()).isEqualTo(dict.keySet());
  }
}
