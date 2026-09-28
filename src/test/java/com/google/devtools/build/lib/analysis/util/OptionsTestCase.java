// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.util;

import static com.google.common.truth.Truth.assertThat;
import static java.util.Arrays.stream;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.common.options.OptionsParser;
import java.util.List;

/** A base class for testing cacheKey related functionality of Option classes. */
public abstract class OptionsTestCase<T extends FragmentOptions> {

  protected abstract Class<T> getOptionsClass();

  /** Construct options parsing the given arguments. */
  protected T create(List<String> args) throws Exception {
    Class<T> cls = getOptionsClass();
    OptionsParser parser = OptionsParser.builder().optionsClasses(ImmutableList.of(cls)).build();
    parser.parse(args);
    return parser.getOptions(cls);
  }

  /**
   * Useful for options which are specified multiple times on the command line. {@code
   * createWithPrefix("--abc=", "x", "y", "z")} is equivalent to {@code create("--abc=x", "--abc=y",
   * "--abc=z")}
   */
  protected T createWithPrefix(String prefix, String... args) throws Exception {
    return createWithPrefix(ImmutableList.of(), prefix, args);
  }

  /**
   * Variant of {@link #createWithPrefix(String, String...)} with additional fixed set of options.
   */
  protected T createWithPrefix(ImmutableList<String> fixed, String prefix, String... args)
      throws Exception {
    ImmutableList.Builder<String> builder = ImmutableList.builder();
    builder.addAll(fixed);
    stream(args).map(x -> prefix + x).forEach(builder::add);
    return create(builder.build());
  }

  protected void assertSame(T one, T two) {
    // We normalize first, since that is what BuildOptions.checkSum() does.
    // We do not use BuildOptions.checkSum() because in case of test failure,
    // the diff on cacheKey is humanreadable.
    FragmentOptions oneNormalized = one.getNormalized();
    FragmentOptions twoNormalized = two.getNormalized();
    assertThat(oneNormalized.cacheKey()).isEqualTo(twoNormalized.cacheKey());
    // Also check equality of toString() as that influences the ST-hash computation.
    assertThat(oneNormalized.toString()).isEqualTo(twoNormalized.toString());
  }

  protected void assertDifferent(T one, T two) {
    // We normalize first, since that is what BuildOptions.checkSum() does.
    // We do not use BuildOptions.checkSum() because in case of test failure,
    // the diff on cacheKey is humanreadable.
    FragmentOptions oneNormalized = one.getNormalized();
    FragmentOptions twoNormalized = two.getNormalized();
    assertThat(oneNormalized.cacheKey()).isNotEqualTo(twoNormalized.cacheKey());
    // Also check equality of toString() as that influences the ST-hash computation.
    assertThat(oneNormalized.toString()).isNotEqualTo(twoNormalized.toString());
  }
}
