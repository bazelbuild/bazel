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

/// Object that displays a greeting.
pub struct Greeter {
    greeting: String,
}

/// Implementation of Greeter.
impl Greeter {
    /// Constructs a new `Greeter`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hello_lib::greeter;
    ///
    /// let greeter = Greeter::new("Hello");
    /// ```
    pub fn new(greeting: &str) -> Greeter {
        Greeter { greeting: greeting.to_string(), }
    }

    /// Returns the greeting as a string.
    ///
    /// # Examples
    ///
    /// ```
    /// use hello_lib::greeter;
    ///
    /// let greeter = Greeter::new("Hello");
    /// let greeting = greeter.greeting("World");
    /// ```
    pub fn greeting(&self, thing: &str) -> String {
        format!("{} {}", &self.greeting, thing)
    }

    /// Prints the greeting.
    ///
    /// # Examples
    ///
    /// ```
    /// use hello_lib::greeter;
    ///
    /// let greeter = Greeter::new("Hello");
    /// greeter.greet("World");
    /// ```
    pub fn greet(&self, thing: &str) {
        println!("{} {}", &self.greeting, thing);
    }
}
