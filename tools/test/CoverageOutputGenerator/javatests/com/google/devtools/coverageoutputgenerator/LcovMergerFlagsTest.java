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

package com.google.devtools.coverageoutputgenerator;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;

public class LcovMergerFlagsTest {
  @Test
  public void parseFlagsTestCoverageDirOutputFile() {
    LcovMergerFlags flags =
        LcovMergerFlags.parseFlags(
            new String[] {
              "--coverage_dir=my_dir", "--output_file=my_file",
            });
    assertThat(flags.coverageDir()).isEqualTo("my_dir");
    assertThat(flags.outputFile()).isEqualTo("my_file");
    assertThat(flags.reportsFile()).isNull();
    assertThat(flags.filterSources()).isEmpty();
  }

  @Test
  public void parseFlagsTestReportsFileOutputFile() {
    LcovMergerFlags flags =
        LcovMergerFlags.parseFlags(
            new String[] {
              "--reports_file=my_reports_file", "--output_file=my_file",
            });
    assertThat(flags.reportsFile()).isEqualTo("my_reports_file");
    assertThat(flags.outputFile()).isEqualTo("my_file");
    assertThat(flags.coverageDir()).isNull();
    assertThat(flags.filterSources()).isEmpty();
  }

  @Test
  public void parseFlagsTestReportsFileOutputFileFilterSources() {
    LcovMergerFlags flags =
        LcovMergerFlags.parseFlags(
            new String[] {
              "--reports_file=my_reports_file",
              "--output_file=my_file",
              "--filter_sources=first_filter"
            });
    assertThat(flags.reportsFile()).isEqualTo("my_reports_file");
    assertThat(flags.outputFile()).isEqualTo("my_file");
    assertThat(flags.coverageDir()).isNull();
    assertThat(flags.filterSources()).containsExactly("first_filter");
  }

  @Test
  public void parseFlagsTestReportsFileOutputFileMultipleFilterSources() {
    LcovMergerFlags flags =
        LcovMergerFlags.parseFlags(
            new String[] {
              "--reports_file=my_reports_file",
              "--output_file=my_file",
              "--filter_sources=first_filter",
              "--filter_sources=second_filter"
            });
    assertThat(flags.reportsFile()).isEqualTo("my_reports_file");
    assertThat(flags.outputFile()).isEqualTo("my_file");
    assertThat(flags.coverageDir()).isNull();
    assertThat(flags.filterSources()).containsExactly("first_filter", "second_filter");
  }

  @Test(expected = IllegalArgumentException.class)
  public void parseFlagsTestCoverageDirAndReportsFile() {
    LcovMergerFlags.parseFlags(
        new String[] {"--reports_file=my_reports_file", "--coverage_dir=my_coverage_dir"});
  }

  @Test(expected = IllegalArgumentException.class)
  public void parseFlagsTestEmptyFlags() {
    LcovMergerFlags.parseFlags(new String[] {});
  }

  @Test(expected = IllegalArgumentException.class)
  public void parseFlagsTestNoOutputFile() {
    LcovMergerFlags.parseFlags(
        new String[] {
          "--reports_file=my_reports_file",
        });
  }

  @Test(expected = IllegalArgumentException.class)
  public void parseFlagsTestUnknownFlag() {
    LcovMergerFlags.parseFlags(
        new String[] {
          "--fake_flag=my_reports_file",
        });
  }

  @Test(expected = IllegalArgumentException.class)
  public void parseFlagsTestInvalidFlagValue() {
    LcovMergerFlags.parseFlags(
        new String[] {
          "--reports_file", "--output_file=my_file",
        });
  }

  @Test(expected = IllegalArgumentException.class)
  public void parseFlagsTestInvalidFlagValueWithoutDashes() {
    LcovMergerFlags.parseFlags(
        new String[] {
          "reports_file", "--output_file=my_file",
        });
  }
}
