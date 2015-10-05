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

/// Returns the nth Fibonacci number.
///
/// # Examples
///
/// ```
/// fibonacci::fibonacci(5)
/// ```
pub fn fibonacci(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    let mut n1: u64 = 0;
    let mut n2: u64 = 1;
    for _ in 1..n {
        let sum = n1 + n2;
        n1 = n2;
        n2 = sum;
    }
    n2
}

#[cfg(test)]
mod test {
    use super::fibonacci;

    #[test]
    fn test_fibonacci() {
        let numbers : Vec<u64> =
            vec![0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144];
        for (i, number) in numbers.iter().enumerate() {
            assert_eq!(*number, fibonacci(i as u64));
        }
    }
}
