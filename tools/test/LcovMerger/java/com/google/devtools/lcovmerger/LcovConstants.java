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

package com.google.devtools.lcovmerger;

/**
 * Stores markers used by the lcov tracefile. See
 * <a href="http://ltp.sourceforge.net/coverage/lcov/geninfo.1.php"> lcov documentation</a>
 */
class LcovConstants {
  static final String SF_MARKER = "SF:";
  static final String FN_MARKER = "FN:";
  static final String FNDA_MARKER = "FNDA:";
  static final String FNF_MARKER = "FNF:";
  static final String FNH_MARKER = "FNH:";
  static final String BRDA_MARKER = "BRDA:";
  static final String BA_MARKER = "BA:";
  static final String BRF_MARKER = "BRF:";
  static final String BRH_MARKER = "BRH:";
  static final String DA_MARKER = "DA:";
  static final String LH_MARKER = "LH:";
  static final String LF_MARKER = "LF:";
  static final String END_OF_RECORD_MARKER = "end_of_record";
  static final String LCOV_DELIMITER = ",";
  static final String TAKEN = "-";
  static final String TRACEFILE_EXTENSION = ".dat";
}
