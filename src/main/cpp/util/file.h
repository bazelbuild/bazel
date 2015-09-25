// Copyright 2014 The Bazel Authors. All rights reserved.
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
#ifndef BAZEL_SRC_MAIN_CPP_UTIL_FILE_H_
#define BAZEL_SRC_MAIN_CPP_UTIL_FILE_H_

#include <string>

namespace blaze_util {

using std::string;

// Returns the part of the path before the final "/".  If there is a single
// leading "/" in the path, the result will be the leading "/".  If there is
// no "/" in the path, the result is the empty prefix of the input (i.e., "").
string Dirname(const string &path);

// Returns the part of the path after the final "/".  If there is no
// "/" in the path, the result is the same as the input.
string Basename(const string &path);

string JoinPath(const string &path1, const string &path2);

// Checks each element of the PATH variable for executable. If none is found, ""
// is returned.  Otherwise, the full path to executable is returned. Can die if
// looking up PATH fails.
string Which(const string &executable);

}  // namespace blaze_util

#endif  // BAZEL_SRC_MAIN_CPP_UTIL_FILE_H_
