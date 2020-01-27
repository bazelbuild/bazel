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

#ifndef BAZEL_SRC_MAIN_CPP_UTIL_PROFILER_H_
#define BAZEL_SRC_MAIN_CPP_UTIL_PROFILER_H_

#include <stdint.h>  // int64_t
#include <stdio.h>   // printf

namespace blaze_util {
namespace profiler {

// A time duration, measured in some implementation-dependent units.
//
// Using a struct to wrap int64_t yields a unique type that we can't
// accidentally use as a type-synonym of other int64_t types.
struct Ticks {
  int64_t value_;
  static Ticks Now();
};

// A time duration, measured in microseconds.
//
// Using a struct to wrap int64_t yields a unique type that we can't
// accidentally use as a type-synonym of other int64_t types.
struct Duration {
  int64_t micros_;
  static Duration FromTicks(const Ticks ticks);
};

// Accumulates statistics about a function or some C++ scope.
//
// Usage: see ScopedTask.
//
// Records the number of times the scope was entered (the function called) and
// the total time spent in there. Prints the statistics in the destructor.
class Task {
  const char* name_;
  uint64_t calls_;
  Ticks total_;

 public:
  Task(const char* name) : name_(name), calls_(0), total_({0}) {}
  ~Task();
  void AddCall() { calls_++; }
  void AddTicks(const Ticks t) { total_.value_ += t.value_; }
  uint64_t GetCalls() const { return calls_; }
  Duration GetDuration() const { return Duration::FromTicks(total_); }
};

// Measures elapsed time.
//
// Example:
//   void foo() {
//     StopWatch s;
//     ...
//     s.PrintAndReset("marker 1");  // prints elapsed time since creation
//     ...
//     s.PrintAndReset("marker 2");  // prints elapsed time since "marker 1"
//     ...
//     s.Reset();
//     ...
//     Ticks t1 = s.Elapsed();  // time since Reset
//     ...
//     Ticks t2 = s.Elapsed();  // time since Reset, not since t1
//   }
//
class StopWatch {
  Ticks start_;

 public:
  // Constructor -- it also resets this StopWatch.
  StopWatch() { start_ = Ticks::Now(); }

  // Prints elapsed time since last reset, then resets.
  //
  // Args:
  //   name: a descriptive name, will be printed in the output
  void PrintAndReset(const char* name);

  // Returns the elapsed time since the last reset as `Ticks`.
  Ticks Elapsed() const {
    Ticks now = Ticks::Now();
    return {now.value_ - start_.value_};
  }

  // Returns the elapsed time since the last reset as `Duration`.
  Duration ElapsedTime() const { return Duration::FromTicks(Elapsed()); }

  // Resets this StopWatch to the current time.
  void Reset() { start_ = Ticks::Now(); }
};

// Measures the execution duration of a given C++ scope.
//
// The constructor records one call of the scope in a `Task` object, and the
// destructor records the time spent in the scope in that `Task` object.
//
// Usage:
//   create one Task that accumulates the statistics for a given function
//   or scope, and create one ScopedTask in the beginning of the scope you want
//   to measure. Every time the scope is entered (the function is called), a
//   ScopedTask is created, then destructed when the execution leaves the scope.
//   The destructor then records the statistics in the Task.
//
// Example:
//   Task slow_function_stats("slow function");  // d'tor prints stats
//
//   void slow_function() {
//     ScopedTask prof(&slow_function_stats);
//     ...
//   }
//
class ScopedTask {
 public:
  ScopedTask(Task* s) : stat_(s) { stat_->AddCall(); }
  ~ScopedTask() { stat_->AddTicks(prof_.Elapsed()); }

 private:
  Task* stat_;
  StopWatch prof_;
};

}  // namespace profiler
}  // namespace blaze_util

#endif  // BAZEL_SRC_MAIN_CPP_UTIL_PROFILER_H_
