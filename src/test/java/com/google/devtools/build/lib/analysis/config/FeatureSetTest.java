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

package com.google.devtools.build.lib.analysis.config;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.analysis.config.FeatureSet.merge;

import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class FeatureSetTest {
  private static FeatureSet parse(String... features) {
    return FeatureSet.parse(Arrays.asList(features));
  }

  @Test
  public void parse_basic() {
    FeatureSet featureSet = parse("foo", "bar", "-baz");
    assertThat(featureSet.on()).containsExactly("foo", "bar");
    assertThat(featureSet.off()).containsExactly("baz");
  }

  @Test
  public void parse_offTrumpsOn() {
    FeatureSet featureSet = parse("foo", "bar", "-bar");
    assertThat(featureSet.on()).containsExactly("foo");
    assertThat(featureSet.off()).containsExactly("bar");
  }

  @Test
  public void parse_noLayeringCheck() {
    FeatureSet featureSet = parse("foo", "bar", "no_layering_check");
    assertThat(featureSet.on()).containsExactly("foo", "bar");
    assertThat(featureSet.off()).containsExactly("layering_check");
  }

  @Test
  public void merge_basic() {
    assertThat(merge(parse("foo", "-bar"), parse("kek", "-lol")))
        .isEqualTo(parse("foo", "kek", "-bar", "-lol"));
  }

  @Test
  public void merge_fineTrumpsCoarse() {
    assertThat(merge(parse("foo", "-bar"), parse("bar", "-foo"))).isEqualTo(parse("-foo", "bar"));
  }

  @Test
  public void merge_associativity() {
    FeatureSet a = parse("foo", "bar", "-baz");
    FeatureSet b = parse("-foo", "-bar");
    FeatureSet c = parse("foo", "baz");
    assertThat(merge(a, merge(b, c))).isEqualTo(merge(merge(a, b), c));
    assertThat(merge(b, merge(c, a))).isEqualTo(merge(merge(b, c), a));
    assertThat(merge(c, merge(b, a))).isEqualTo(merge(merge(c, b), a));
  }

  @Test
  public void mergeWithGlobalFeatures_basic() {
    assertThat(FeatureSet.mergeWithGlobalFeatures(parse("foo", "-bar"), parse("kek", "-lol")))
        .isEqualTo(parse("foo", "kek", "-bar", "-lol"));
  }

  @Test
  public void mergeWithGlobalFeatures_globalOffTrumpsEverything() {
    assertThat(FeatureSet.mergeWithGlobalFeatures(parse("foo", "-bar"), parse("bar", "-foo")))
        .isEqualTo(parse("-bar", "-foo"));
  }
}
