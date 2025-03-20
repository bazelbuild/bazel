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
package com.google.devtools.build.lib.analysis.config;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.testing.GcFinalization;
import com.google.devtools.common.options.OptionsParsingException;
import java.lang.ref.WeakReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BuildOptionsCache}. */
@RunWith(JUnit4.class)
public final class BuildOptionsCacheTest {

  private final BuildOptionsCache<Context> cache =
      new BuildOptionsCache<>(
          (options, context, unused) -> {
            BuildOptionsView clone = options.clone();
            clone.get(CoreOptions.class).cpu = context.val;
            return clone.underlying();
          });

  @Test
  public void appliesTransitionFunction() throws Exception {
    BuildOptionsView from = createOptions("--cpu=default");
    BuildOptions to = cache.applyTransition(from, new Context("abc"), null);
    assertCpu(from.underlying(), "default"); // No change.
    assertCpu(to, "abc");
  }

  @Test
  public void cachesTransition() throws Exception {
    BuildOptions to1 =
        cache.applyTransition(createOptions("--cpu=default"), new Context("abc"), null);
    BuildOptions to2 =
        cache.applyTransition(createOptions("--cpu=default"), new Context("abc"), null);
    assertThat(to2).isSameInstanceAs(to1);
  }

  @Test
  public void cacheKeyRespectsFromOptions() throws Exception {
    BuildOptions to1 =
        cache.applyTransition(
            createOptions("--cpu=default", "--host_cpu=one"), new Context("abc"), null);
    BuildOptions to2 =
        cache.applyTransition(
            createOptions("--cpu=default", "--host_cpu=two"), new Context("abc"), null);
    assertCpu(to1, "abc");
    assertCpu(to2, "abc");
    assertHostCpu(to1, "one");
    assertHostCpu(to2, "two");
  }

  @Test
  public void cacheKeyRespectsContext() throws Exception {
    BuildOptions to1 =
        cache.applyTransition(createOptions("--cpu=default"), new Context("abc"), null);
    BuildOptions to2 =
        cache.applyTransition(createOptions("--cpu=default"), new Context("xyz"), null);
    assertCpu(to1, "abc");
    assertCpu(to2, "xyz");
  }

  // We would like to also test that the toOptions are not strongly retained, but since they are
  // referenced softly, this is not easy to do.
  @Test
  public void doesNotRetainFromOptions() throws Exception {
    BuildOptionsView from = createOptions("--cpu=default");
    var unused = cache.applyTransition(from, new Context("abc"), null);
    WeakReference<BuildOptions> fromRef = new WeakReference<>(from.underlying());
    from = null;
    GcFinalization.awaitClear(fromRef);
  }

  private static BuildOptionsView createOptions(String... args) throws OptionsParsingException {
    return new BuildOptionsView(
        BuildOptions.of(ImmutableList.of(CoreOptions.class), args),
        ImmutableSet.of(CoreOptions.class));
  }

  private static void assertCpu(BuildOptions options, String expected) {
    assertThat(options.get(CoreOptions.class).cpu).isEqualTo(expected);
  }

  private static void assertHostCpu(BuildOptions options, String expected) {
    assertThat(options.get(CoreOptions.class).hostCpu).isEqualTo(expected);
  }

  /** Simple value class for testing the context parameter. */
  private static final class Context {
    private final String val;

    Context(String val) {
      this.val = val;
    }

    @Override
    public int hashCode() {
      return val.hashCode();
    }

    @Override
    public boolean equals(Object o) {
      if (o == this) {
        return true;
      }
      return o instanceof Context && val.equals(((Context) o).val);
    }
  }
}
