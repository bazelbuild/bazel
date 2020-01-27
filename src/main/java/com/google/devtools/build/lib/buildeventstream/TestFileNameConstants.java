// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.buildeventstream;

/**
 * Class providing constants for naming files in associated with tests.
 *
 * <p>The file names associated with a test are indexed in the build-event protocol by a string in
 * order to allow extensions of bazel to add their own files. This class provides constants for the
 * names of the standard files associated with a test.
 */
public class TestFileNameConstants {
  public static final String SPLIT_LOGS = "test.splitlogs";
  public static final String TEST_INFRASTRUCTURE_FAILURE = "test.infrastructure_failure";
  public static final String TEST_LOG = "test.log";
  public static final String TEST_STDERR = "test.stderr";
  public static final String TEST_WARNINGS = "test.warnings";
  public static final String TEST_XML = "test.xml";
  public static final String UNDECLARED_OUTPUTS_ANNOTATIONS = "test.outputs_manifest__ANNOTATIONS";
  public static final String UNDECLARED_OUTPUTS_MANIFEST = "test.outputs_manifest__MANIFEST";
  public static final String UNDECLARED_OUTPUTS_ZIP = "test.outputs__outputs.zip";
  public static final String UNUSED_RUNFILES_LOG = "test.unused_runfiles_log";
  public static final String TEST_COVERAGE = "test.lcov";
  public static final String BASELINE_COVERAGE = "baseline.lcov";
}
