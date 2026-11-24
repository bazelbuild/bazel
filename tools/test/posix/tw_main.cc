// Copyright 2026 The Bazel Authors. All rights reserved.
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

#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

namespace bazel {
namespace tools {
namespace test_wrapper {

int TestWrapperMain(int argc, char** argv);

}  // namespace test_wrapper
}  // namespace tools
}  // namespace bazel

namespace {

std::string GetEnv(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? "" : value;
}

// Read one XML character, replacing malformed UTF-8 and XML 1.0-disallowed
// code points with '?'. An invalid continuation byte is left for the next read.
std::string_view ReadXmlCharacter(int first, std::istream& in, char (&buf)[4]) {
  buf[0] = static_cast<char>(first);
  if (first < 0x80) {
    return first >= 0x20 || first == '\t' || first == '\n' || first == '\r'
               ? std::string_view(buf, 1)
               : "?";
  }
  if (first < 0xc2 || first > 0xf4) {
    return "?";
  }
  int length = first < 0xe0 ? 2 : first < 0xf0 ? 3 : 4;
  int codepoint = first & (0x7f >> length);
  for (int i = 1; i < length; ++i) {
    int byte = in.peek();
    if (byte < 0x80 || byte > 0xbf ||
        (i == 1 && ((first == 0xe0 && byte < 0xa0) ||
                    (first == 0xed && byte >= 0xa0) ||
                    (first == 0xf0 && byte < 0x90) ||
                    (first == 0xf4 && byte >= 0x90)))) {
      return "?";
    }
    buf[i] = static_cast<char>(in.get());
    codepoint = (codepoint << 6) | (byte & 0x3f);
  }
  return codepoint == 0xfffe || codepoint == 0xffff
             ? "?"
             : std::string_view(buf, length);
}

void WriteLog(std::istream& in, std::ostream& out) {
  int brackets = 0;
  size_t newlines = 0;
  for (int byte; (byte = in.get()) != std::char_traits<char>::eof();) {
    if (byte == '\n') {
      ++newlines;
      brackets = 0;
      continue;
    }
    // Trim trailing newlines as generate-xml.sh does, without buffering the log.
    for (; newlines > 0; --newlines) {
      out.put('\n');
    }
    if (byte == '>' && brackets == 2) {
      out << ">]]<![CDATA[>";
    } else {
      char buf[4];
      out << ReadXmlCharacter(byte, in, buf);
    }
    brackets = byte == ']' ? (brackets == 0 ? 1 : 2) : 0;
  }
}

void WriteAttribute(std::ostream& out, const std::string& value) {
  std::istringstream in(value);
  for (int ch; (ch = in.get()) != std::char_traits<char>::eof();) {
    switch (ch) {
      case '&':
        out << "&amp;";
        break;
      case '<':
        out << "&lt;";
        break;
      case '"':
        out << "&quot;";
        break;
      default:
        char buf[4];
        out << ReadXmlCharacter(ch, in, buf);
    }
  }
}

int GenerateXml(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: posix_tw --generate-xml LOG XML DURATION EXIT_CODE\n";
    return 2;
  }
  const char* duration = argv[4];
  const char* exit_code = argv[5];
  const bool failed = std::string_view(exit_code) != "0";

  std::string name = GetEnv("TEST_BINARY");
  if (name.compare(0, 2, "./") == 0) {
    name.erase(0, 2);
  }
  if (name.compare(0, 3, "../") == 0) {
    name.erase(0, 3);
  }
  std::string total_shards = GetEnv("TEST_TOTAL_SHARDS");
  if (!total_shards.empty() && total_shards != "0") {
    std::string index = GetEnv("TEST_SHARD_INDEX");
    unsigned int shard_index = 0;
    std::from_chars(index.data(), index.data() + index.size(), shard_index);
    name += "_shard_" + std::to_string(1ULL + shard_index) + "/" + total_shards;
  }

  std::ifstream in(argv[2], std::ios::binary);
  // generate-xml.sh also writes an empty system-out when the log is missing.
  if (!in && errno != ENOENT) {
    std::cerr << "ERROR: could not open test log " << argv[2] << '\n';
    return 1;
  }
  std::ofstream out(argv[3], std::ios::binary);
  if (out) {
    out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
           "<testsuites>\n  <testsuite name=\"";
    WriteAttribute(out, name);
    out << "\" tests=\"1\" failures=\"0\" errors=\"" << (failed ? 1 : 0)
        << "\">\n    <testcase name=\"";
    WriteAttribute(out, name);
    out << "\" classname=\"\" status=\"run\" duration=\"" << duration
        << "\" time=\"" << duration << "\">";
    if (failed) {
      out << "<error message=\"exited with error code " << exit_code
          << "\"></error>";
    }
    out << "</testcase>\n      <system-out>\n"
           "Generated test.log (if the file is not UTF-8, then this may be "
           "unreadable):\n<![CDATA[";
    WriteLog(in, out);
    out << "]]>\n      </system-out>\n    </testsuite>\n</testsuites>\n";
    out.close();
    if (!in.bad() && out) {
      return 0;
    }
  }
  std::cerr << "ERROR: could not generate test XML from " << argv[2] << " to "
            << argv[3] << '\n';
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string_view(argv[1]) == "--generate-xml") {
    return GenerateXml(argc, argv);
  }
  return bazel::tools::test_wrapper::TestWrapperMain(argc, argv);
}
