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

#include "tools/cpp/modules_tools/common/json.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <string>
#include <vector>

TEST(JsonTest, EncodeTest) {
  Json &json = Json::instance();

  ASSERT_EQ(json.encode(nullptr), "null");
  ASSERT_EQ(json.encode(true), "true");
  ASSERT_EQ(json.encode(false), "false");
  ASSERT_EQ(json.encode(-123), "-123");

  // This implementation don't support large number
  // ASSERT_EQ(json.encode(12345 * 12345 * 12345 * 12345 * 12345 * 12345),
  // "3539537889086624823140625");
  // ASSERT_EQ(json.encode(static_cast<double>(12345 * 12345 * 12345 * 12345 *
  // 12345 * 12345)), "3.539537889086625e+24");
  ASSERT_EQ(json.encode(12.345e67), "1.2345e+68");
  ASSERT_EQ(json.encode("hello"), "\"hello\"");
  ASSERT_EQ(json.encode("\t"), "\"\\t\"");
  ASSERT_EQ(json.encode("\r"), "\"\\r\"");
  ASSERT_EQ(json.encode("\n"), "\"\\n\"");
  ASSERT_EQ(json.encode("'"), "\"'\"");
  ASSERT_EQ(json.encode("\""), "\"\\\"\"");
  ASSERT_EQ(json.encode("/"), "\"/\"");
  ASSERT_EQ(json.encode("\\"), "\"\\\\\"");
  ASSERT_EQ(json.encode(""), "\"\"");
  JsonValue::ArrayType arr = {JsonValue(1), JsonValue(2), JsonValue(3)};
  ASSERT_EQ(json.encode(arr), "[1,2,3]");

  // Mapping of key-values for mapping JSON of objects/dictionaries
  JsonValue::ObjectType m = {{"x", JsonValue(1)}, {"y", JsonValue("two")}};
  ASSERT_EQ(json.encode(m), "{\"x\":1,\"y\":\"two\"}");
}

TEST(JsonTest, DecodeTest) {
  Json &json = Json::instance();

  ASSERT_EQ(json.decode("null"), JsonValue(nullptr));
  ASSERT_EQ(json.decode("true"), JsonValue(true));
  ASSERT_EQ(json.decode("false"), JsonValue(false));
  ASSERT_EQ(json.decode("-123"), JsonValue(-123));
  ASSERT_EQ(json.decode("-0"), JsonValue(0));
  // This implementation don't support large number
  // ASSERT_EQ(json.decode("3539537889086624823140625"),
  // JsonValue(3539537889086624823140625ll));
  // ASSERT_EQ(json.decode("3539537889086624823140625.0"),
  // JsonValue(static_cast<double>(3539537889086624823140625ll)));

  // Additional decoding examples
  ASSERT_EQ(json.decode("[]").as_array().size(), 0);
  ASSERT_EQ(json.decode("[1]").as_array().size(), 1);
  auto arr = json.decode("[1, 2, 3]").as_array();
  ASSERT_EQ(arr.size(), 3);
  ASSERT_EQ(arr[0].as_long(), 1);
  ASSERT_EQ(arr[1].as_long(), 2);
  ASSERT_EQ(arr[2].as_long(), 3);

  auto obj = json.decode("{\"one\": 1, \"two\": 2}").as_object();
  ASSERT_EQ(obj.size(), 2);
  ASSERT_EQ(obj["one"].as_long(), 1);
  ASSERT_EQ(obj["two"].as_long(), 2);
}
