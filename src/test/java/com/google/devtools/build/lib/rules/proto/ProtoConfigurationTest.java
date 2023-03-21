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

package com.google.devtools.build.lib.rules.proto;

import com.google.devtools.build.lib.analysis.util.OptionsTestCase;
import com.google.devtools.build.lib.rules.proto.ProtoConfiguration.Options;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class ProtoConfigurationTest extends OptionsTestCase<Options> {

  private static final String HDR_SUFFIXES_PREFIX = "--cc_proto_library_header_suffixes=";
  private static final String SRC_SUFFIXES_PREFIX = "--cc_proto_library_source_suffixes=";

  @Override
  protected Class<Options> getOptionsClass() {
    return Options.class;
  }

  @Test
  public void testHdrSuffixes_ordering() throws Exception {
    Options one = createWithPrefix(HDR_SUFFIXES_PREFIX, ".one.h,.two.h");
    Options two = createWithPrefix(HDR_SUFFIXES_PREFIX, ".two.h,.one.h");
    assertSame(one, two);
  }

  @Test
  public void testHdrSuffixes_duplicates() throws Exception {
    Options one = createWithPrefix(HDR_SUFFIXES_PREFIX, ".one.h,.one.h");
    Options two = createWithPrefix(HDR_SUFFIXES_PREFIX, ".one.h");
    assertSame(one, two);
  }

  @Test
  public void testSrcSuffixes_ordering() throws Exception {
    Options one = createWithPrefix(SRC_SUFFIXES_PREFIX, ".one.cc,.two.cc");
    Options two = createWithPrefix(SRC_SUFFIXES_PREFIX, ".two.cc,.one.cc");
    assertSame(one, two);
  }

  @Test
  public void testSrcSuffixes_duplicates() throws Exception {
    Options one = createWithPrefix(SRC_SUFFIXES_PREFIX, ".one.cc,.one.cc");
    Options two = createWithPrefix(SRC_SUFFIXES_PREFIX, ".one.cc");
    assertSame(one, two);
  }
}
