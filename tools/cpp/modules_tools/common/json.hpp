// Copyright 2024 The Bazel Authors. All rights reserved.
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

#pragma once

#include <algorithm>
#include <any>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>
// forward decl
struct JsonValue;
inline std::string to_json(const JsonValue &data);

// Define a basic struct for JSON values
struct JsonValue {
  using ObjectType = std::map<std::string, JsonValue>;
  using ArrayType = std::vector<JsonValue>;

  std::variant<std::string, bool, long, double, ObjectType, ArrayType,
               std::nullptr_t>
      value;

  JsonValue() : value(nullptr) {}
  JsonValue(const std::string &v) : value(v) {}
  JsonValue(const char *v) : value(std::string(v)) {}
  JsonValue(bool v) : value(v) {}
  JsonValue(long v) : value(v) {}
  JsonValue(int v) : value((long)v) {}
  JsonValue(double v) : value(v) {}
  JsonValue(const ObjectType &v) : value(v) {}
  JsonValue(const ArrayType &v) : value(v) {}
  JsonValue(std::nullptr_t) : value(nullptr) {}

  bool is_null() const { return std::holds_alternative<std::nullptr_t>(value); }
  bool is_string() const { return std::holds_alternative<std::string>(value); }
  bool is_object() const { return std::holds_alternative<ObjectType>(value); }
  bool is_array() const { return std::holds_alternative<ArrayType>(value); }
  bool is_bool() const { return std::holds_alternative<bool>(value); }
  bool is_long() const { return std::holds_alternative<long>(value); }
  bool is_double() const { return std::holds_alternative<double>(value); }

  const std::string &as_string() const { return std::get<std::string>(value); }
  const ObjectType &as_object() const { return std::get<ObjectType>(value); }
  const ArrayType &as_array() const { return std::get<ArrayType>(value); }
  bool as_bool() const { return std::get<bool>(value); }
  long as_long() const { return std::get<long>(value); }
  double as_double() const { return std::get<double>(value); }

  // Implement equality operator
  bool operator==(const JsonValue &other) const {
    if (value.index() != other.value.index())
      return false;

    if (is_null())
      return true;
    if (is_string())
      return as_string() == other.as_string();
    if (is_bool())
      return as_bool() == other.as_bool();
    if (is_long())
      return as_long() == other.as_long();
    if (is_double())
      return as_double() == other.as_double();
    if (is_object())
      return as_object() == other.as_object();
    if (is_array())
      return as_array() == other.as_array();

    return false;
  }

  bool operator!=(const JsonValue &other) const { return !(*this == other); }

  std::string dump() const { return to_json(*this); }
};

// Define the JSON parser class
class Json {
public:
  // Singleton instance
  static Json &instance() {
    static Json INSTANCE;
    return INSTANCE;
  }

  // Function to encode an object to JSON
  std::string encode(const JsonValue &x) const {
    Encoder enc;
    try {
      enc.encode(x);
    } catch (const std::overflow_error &e) {
      throw std::runtime_error("nesting depth limit exceeded");
    }
    return enc.out.str();
  }

  // Function to decode a JSON string to an object
  JsonValue decode(const std::string &x) const {
    try {
      return Decoder(x).decode();
    } catch (const std::runtime_error &e) {
      throw std::runtime_error("Invalid JSON string");
    }
  }

private:
  Json() {}

  // Encoder class to serialize objects to JSON
  class Encoder {
  public:
    void encode(const JsonValue &x) {
      if (x.is_null()) {
        out << "null";
        return;
      }
      if (x.is_string()) {
        append_quoted(x.as_string());
        return;
      }
      if (x.is_bool()) {
        out << std::boolalpha << x.as_bool();
        return;
      }
      if (x.is_long()) {
        out << x.as_long();
        return;
      }
      if (x.is_double()) {
        double d = x.as_double();
        if (!std::isfinite(d)) {
          throw std::runtime_error("Cannot encode non-finite float");
        }
        out << std::setprecision(std::numeric_limits<double>::digits10) << d;
        return;
      }
      if (x.is_object()) {
        const auto &m = x.as_object();
        out << '{';
        std::string sep = "";
        for (const auto &item : m) {
          out << sep;
          sep = ",";
          append_quoted(item.first);
          out << ':';
          encode(item.second);
        }
        out << '}';
        return;
      }
      if (x.is_array()) {
        const auto &v = x.as_array();
        out << '[';
        std::string sep = "";
        for (const auto &value : v) {
          out << sep;
          sep = ",";
          encode(value);
        }
        out << ']';
        return;
      }
      // Add more cases for other types as needed
      throw std::runtime_error("Cannot encode value as JSON");
    }

    std::ostringstream out;

  private:
    void append_quoted(const std::string &s) {
      out << '"';
      for (char c : s) {
        switch (c) {
        case '"':
          out << "\\\"";
          break;
        case '\\':
          out << "\\\\";
          break;
        case '\b':
          out << "\\b";
          break;
        case '\f':
          out << "\\f";
          break;
        case '\n':
          out << "\\n";
          break;
        case '\r':
          out << "\\r";
          break;
        case '\t':
          out << "\\t";
          break;
        default:
          if ('\x00' <= c && c <= '\x1f') {
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                << static_cast<long>(c);
          } else {
            out << c;
          }
        }
      }
      out << '"';
    }
  };

  // Decoder class to parse JSON strings to objects
  class Decoder {
  public:
    Decoder(const std::string &s) : s(s), i(0) {}

    JsonValue decode() {
      auto x = parse();
      if (skip_space()) {
        throw std::runtime_error("Unexpected character after value");
      }
      return x;
    }

  private:
    std::string s;
    size_t i;

    JsonValue parse() {
      char c = next();
      switch (c) {
      case '"':
        return parse_string();
      case 'n':
        if (s.substr(i, 4) == "null") {
          i += 4;
          return nullptr;
        }
        break;
      case 't':
        if (s.substr(i, 4) == "true") {
          i += 4;
          return true;
        }
        break;
      case 'f':
        if (s.substr(i, 5) == "false") {
          i += 5;
          return false;
        }
        break;
      case '[':
        return parse_array();
      case '{':
        return parse_object();
      default:
        if (isdigit(c) || c == '-') {
          return parse_number(c);
        }
      }
      throw std::runtime_error("Unexpected character");
    }

    JsonValue parse_string() {
      i++; // skip "
      std::ostringstream str;
      while (i < s.size()) {
        char c = s[i++];
        if (c == '"')
          return str.str();
        if (c == '\\') {
          c = s[i++];
          switch (c) {
          case 'b':
            str << '\b';
            break;
          case 'f':
            str << '\f';
            break;
          case 'n':
            str << '\n';
            break;
          case 'r':
            str << '\r';
            break;
          case 't':
            str << '\t';
            break;
          case 'u':
            // Handle \uXXXX
            str << static_cast<char>(std::stoi(s.substr(i, 4), nullptr, 16));
            i += 4;
            break;
          default:
            str << c;
          }
        } else {
          str << c;
        }
      }
      throw std::runtime_error("Unclosed string literal");
    }

    JsonValue parse_array() {
      JsonValue::ArrayType array;
      i++; // skip [
      if (next() != ']') {
        while (true) {
          array.push_back(parse());
          char c = next();
          if (c != ',') {
            if (c != ']')
              throw std::runtime_error("Expected ',' or ']'");
            break;
          }
          i++; // skip ,
        }
      }
      i++; // skip ]
      return array;
    }

    JsonValue parse_object() {
      JsonValue::ObjectType object;
      i++; // skip {
      if (next() != '}') {
        while (true) {
          std::string key = std::get<std::string>(parse().value);
          if (next() != ':')
            throw std::runtime_error("Expected ':'");
          i++; // skip :
          object[key] = parse();
          char c = next();
          if (c != ',') {
            if (c != '}')
              throw std::runtime_error("Expected ',' or '}'");
            break;
          }
          i++; // skip ,
        }
      }
      i++; // skip }
      return object;
    }

    JsonValue parse_number(char c) {
      size_t j = i + 1;
      bool isfloat = false;
      while (j < s.size()) {
        c = s[j];
        if (isdigit(c) || c == '-' || c == '+' || c == '.' || c == 'e' ||
            c == 'E') {
          if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-')
            isfloat = true;
          j++;
        } else {
          break;
        }
      }
      std::string num = s.substr(i, j - i);
      i = j;
      std::istringstream iss(num);
      if (isfloat) {
        double d;
        iss >> d;
        return d;
      } else {
        long n;
        iss >> n;
        return n;
      }
    }

    bool skip_space() {
      while (i < s.size() && isspace(s[i])) {
        i++;
      }
      return i < s.size();
    }

    char next() {
      if (skip_space()) {
        return s[i];
      }
      throw std::runtime_error("Unexpected end of input");
    }
  };
};

inline JsonValue parse_json(const std::string &data) {
  Json &json = Json::instance();
  return json.decode(data);
}

inline std::string to_json(const JsonValue &data) {
  Json &json = Json::instance();
  return json.encode(data);
}
