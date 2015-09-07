// Copyright 2015 Google Inc. All rights reserved.
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

module greeter;

import std.stdio;
import std.string;

/// Displays a greeting.
class Greeter {
 private string greeting;

 public:
  /// Creates a new greeter.
  ///
  /// Params:
  ///     greeting = The greeting to use.
  this(in string greeting) {
    this.greeting = greeting.dup;
  }

  /// Returns the greeting as a string.
  ///
  /// Params:
  ///     thing = The thing to greet
  ///
  /// Returns:
  ///     A greeting as a string.
  string makeGreeting(in immutable string thing) {
    return format("%s %s!", this.greeting, thing);
  }

  /// Prints a greeting.
  ///
  /// Params:
  ///     thing = The thing to greet.
  void greet(in immutable string thing) {
    writeln(makeGreeting(thing));
  }
}
