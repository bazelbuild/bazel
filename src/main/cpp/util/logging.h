// Copyright 2016 The Bazel Authors. All rights reserved.
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
#ifndef BAZEL_SRC_MAIN_CPP_LOGGING_H_
#define BAZEL_SRC_MAIN_CPP_LOGGING_H_

#include <memory>
#include <sstream>
#include <string>

// This file is based off the logging work by the protobuf team in
// stubs/logging.h,
//
// Users of this logging library should use BAZEL_LOG(level) << ""; format,
// and specify how they wish to handle the output of the log messages by
// creating a LogHandler to pass to SetLogHandler().
namespace blaze_util {

enum LogLevel {
  LOGLEVEL_INFO,
  LOGLEVEL_USER,
  LOGLEVEL_WARNING,
  LOGLEVEL_ERROR,
  LOGLEVEL_FATAL,

#ifdef NDEBUG
  LOGLEVEL_DFATAL = LOGLEVEL_ERROR
#else
  LOGLEVEL_DFATAL = LOGLEVEL_FATAL
#endif
};

// Returns a string representation of the log level.
const char* LogLevelName(LogLevel level);

namespace internal {

class LogFinisher;
class LogMessage {
 public:
  LogMessage(LogLevel level, const std::string& filename, int line);
  LogMessage(LogLevel level, const std::string& filename, int line,
             int exit_code);

  LogMessage& operator<<(const std::string& value);
  LogMessage& operator<<(const char* value);
  LogMessage& operator<<(char value);
  LogMessage& operator<<(bool value);
  LogMessage& operator<<(short value);
  LogMessage& operator<<(int value);
  LogMessage& operator<<(unsigned int value);
  LogMessage& operator<<(long value);
  LogMessage& operator<<(unsigned long value);
  LogMessage& operator<<(long long value);
  LogMessage& operator<<(unsigned long long value);
  LogMessage& operator<<(float value);
  LogMessage& operator<<(double value);
  LogMessage& operator<<(long double value);
  LogMessage& operator<<(void* value);

 private:
  friend class LogFinisher;
  void Finish();

  const LogLevel level_;
  const std::string& filename_;
  const int line_;
  // Only used for FATAL log messages.
  const int exit_code_;
  std::stringstream message_;
};

// Used to make the entire "LOG(BLAH) << etc." expression have a void return
// type and print a newline after each message.
class LogFinisher {
 public:
  void operator=(LogMessage& other);
};

template <typename T>
bool IsOk(T status) {
  return status.ok();
}
template <>
inline bool IsOk(bool status) {
  return status;
}

}  // namespace internal

#define BAZEL_LOG(LEVEL)                                                      \
  ::blaze_util::internal::LogFinisher() = ::blaze_util::internal::LogMessage( \
      ::blaze_util::LOGLEVEL_##LEVEL, __FILE__, __LINE__)
#define BAZEL_LOG_IF(LEVEL, CONDITION) !(CONDITION) ? (void)0 : BAZEL_LOG(LEVEL)
#define BAZEL_DIE(EXIT_CODE)                                                  \
  ::blaze_util::internal::LogFinisher() = ::blaze_util::internal::LogMessage( \
      ::blaze_util::LOGLEVEL_FATAL, __FILE__, __LINE__, EXIT_CODE)

#define BAZEL_CHECK(EXPRESSION) \
  BAZEL_LOG_IF(FATAL, !(EXPRESSION)) << "CHECK failed: " #EXPRESSION ": "
#define BAZEL_CHECK_OK(A) BAZEL_CHECK(::blaze_util::internal::IsOk(A))
#define BAZEL_CHECK_EQ(A, B) BAZEL_CHECK((A) == (B))
#define BAZEL_CHECK_NE(A, B) BAZEL_CHECK((A) != (B))
#define BAZEL_CHECK_LT(A, B) BAZEL_CHECK((A) < (B))
#define BAZEL_CHECK_LE(A, B) BAZEL_CHECK((A) <= (B))
#define BAZEL_CHECK_GT(A, B) BAZEL_CHECK((A) > (B))
#define BAZEL_CHECK_GE(A, B) BAZEL_CHECK((A) >= (B))

#ifdef NDEBUG

#define BAZEL_DLOG(LEVEL) BAZEL_LOG_IF(LEVEL, false)

#define BAZEL_DCHECK(EXPRESSION) \
  while (false) BAZEL_CHECK(EXPRESSION)
#define BAZEL_DCHECK_OK(E) BAZEL_DCHECK(::blaze::internal::IsOk(E))
#define BAZEL_DCHECK_EQ(A, B) BAZEL_DCHECK((A) == (B))
#define BAZEL_DCHECK_NE(A, B) BAZEL_DCHECK((A) != (B))
#define BAZEL_DCHECK_LT(A, B) BAZEL_DCHECK((A) < (B))
#define BAZEL_DCHECK_LE(A, B) BAZEL_DCHECK((A) <= (B))
#define BAZEL_DCHECK_GT(A, B) BAZEL_DCHECK((A) > (B))
#define BAZEL_DCHECK_GE(A, B) BAZEL_DCHECK((A) >= (B))

#else  // NDEBUG

#define BAZEL_DLOG BAZEL_LOG

#define BAZEL_DCHECK BAZEL_CHECK
#define BAZEL_DCHECK_OK BAZEL_CHECK_OK
#define BAZEL_DCHECK_EQ BAZEL_CHECK_EQ
#define BAZEL_DCHECK_NE BAZEL_CHECK_NE
#define BAZEL_DCHECK_LT BAZEL_CHECK_LT
#define BAZEL_DCHECK_LE BAZEL_CHECK_LE
#define BAZEL_DCHECK_GT BAZEL_CHECK_GT
#define BAZEL_DCHECK_GE BAZEL_CHECK_GE

#endif  // !NDEBUG

class LogHandler {
 public:
  virtual ~LogHandler() {}
  virtual void HandleMessage(LogLevel level, const std::string& filename,
                             int line, const std::string& message,
                             int exit_code) = 0;

  virtual void SetOutputStream(std::unique_ptr<std::ostream> output_stream) = 0;
  virtual void SetOutputStreamToStderr() = 0;
};

// Sets the log handler that routes all log messages.
// SetLogHandler is not thread-safe.  You should only call it
// at initialization time, and probably not from library code.
void SetLogHandler(std::unique_ptr<LogHandler> new_handler);

// Set the stream to which all log statements will be sent.
void SetLoggingOutputStream(std::unique_ptr<std::ostream> output_stream);
void SetLoggingOutputStreamToStderr();

}  // namespace blaze_util

#endif  // BAZEL_SRC_MAIN_CPP_LOGGING_H_
