// Copyright 2015 The Bazel Authors. All rights reserved.
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

module native_greeter;

extern (C):

struct NativeGreeter {
  char* greeting;
};

/// Creates a new NativeGreeter.
///
/// Params:
///     greeting = The greeting to use.
///
/// Returns:
///     A pointer to a new NativeGreeting struct.
NativeGreeter* native_greeter_new(const(char)* greeting);

/// Prints a greeting to stdout.
///
/// Params:
///     greeter = The pointer to the NativeGreeter object to use.
///     thing = The thing to greet.
void native_greeter_greet(const(NativeGreeter)* greeter, const(char)* thing);

/// Frees the NativeGreeter.
///
/// Params:
///     greeter = The pointer to the NativeGreeter object to use.
void native_greeter_free(NativeGreeter* greeter);
