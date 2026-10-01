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

#include "tools/jdk/proguard_allowlister_lib.h"

#include <cctype>
#include <cstddef>
#include <fstream>
#include <ios>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace proguard_allowlister {

namespace {

std::string Trim(const std::string& s) {
  size_t start = 0;
  while (start < s.size() &&
         std::isspace(static_cast<unsigned char>(s[start]))) {
    start++;
  }
  size_t end = s.size();
  while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
    end--;
  }
  return s.substr(start, end - start);
}

std::vector<std::string> GetTokens(const std::string& s) {
  std::vector<std::string> tokens;
  std::istringstream iss(s);
  std::string token;
  while (iss >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

}  // namespace

ProguardConfigValidator::ProguardConfigValidator(std::string config_path,
                                                 std::string outconfig_path)
    : config_path_(std::move(config_path)),
      outconfig_path_(std::move(outconfig_path)) {}

bool ProguardConfigValidator::ValidateConfigFileExtension(
    std::string* /*error*/) const {
  // Base validator does not enforce any file extension.
  return true;
}

std::string ProguardConfigValidator::StripComments(const std::string& config) {
  std::string result;
  result.reserve(config.size());
  for (size_t i = 0; i < config.size();) {
    if (config[i] == '#') {
      size_t newline = config.find('\n', i);
      if (newline == std::string::npos) {
        break;
      } else {
        i = newline + 1;
      }
    } else {
      result.push_back(config[i]);
      i++;
    }
  }
  return result;
}

std::vector<std::string> ProguardConfigValidator::SplitArgs(
    const std::string& config) {
  std::vector<std::string> args;
  size_t last_end = 0;

  size_t start_ws = 0;
  while (start_ws < config.size() &&
         std::isspace(static_cast<unsigned char>(config[start_ws]))) {
    start_ws++;
  }
  if (start_ws < config.size() && config[start_ws] == '-') {
    args.push_back(std::string());
    last_end = start_ws + 1;
  }

  for (size_t i = last_end; i < config.size(); ++i) {
    if (config[i] == '\n') {
      size_t j = i + 1;
      while (j < config.size() &&
             std::isspace(static_cast<unsigned char>(config[j]))) {
        j++;
      }
      if (j < config.size() && config[j] == '-') {
        args.push_back(config.substr(last_end, i - last_end));
        last_end = j + 1;
        i = j;
      }
    }
  }
  args.push_back(config.substr(last_end));
  return args;
}

bool ProguardConfigValidator::ValidateArg(const std::string& arg) const {
  static const char* const kValidArgs[] = {
      "keep",         "assumenosideeffects",
      "assumevalues", "adaptresourcefilecontents",
      "if",
  };
  for (const char* valid_arg : kValidArgs) {
    if (arg.rfind(valid_arg, 0) == 0) {
      return true;
    }
  }
  std::vector<std::string> tokens = GetTokens(arg);
  if (!tokens.empty()) {
    if (tokens[0] == "dontnote" || tokens[0] == "dontwarn") {
      if (tokens.size() > 1) {
        return true;
      }
    }
  }
  return false;
}

std::vector<std::string> ProguardConfigValidator::Validate(
    const std::string& config) const {
  std::string stripped = StripComments(config);
  std::vector<std::string> args = SplitArgs(stripped);

  std::vector<std::string> invalid_configs;
  for (const auto& raw_arg : args) {
    std::string arg = Trim(raw_arg);
    if (arg.empty() || ValidateArg(arg)) {
      continue;
    }
    std::vector<std::string> tokens = GetTokens(arg);
    if (!tokens.empty()) {
      invalid_configs.push_back("-" + tokens[0]);
    }
  }
  return invalid_configs;
}

bool ProguardConfigValidator::ValidateAndWriteOutput(std::string* error) {
  if (!ValidateConfigFileExtension(error)) {
    return false;
  }

  std::ifstream input_file(config_path_);
  if (!input_file.is_open()) {
    if (error) {
      *error = "Unable to open proguard config file: " + config_path_;
    }
    return false;
  }
  std::string config_string((std::istreambuf_iterator<char>(input_file)),
                            std::istreambuf_iterator<char>());
  input_file.close();

  std::vector<std::string> invalid_configs = Validate(config_string);
  if (!invalid_configs.empty()) {
    if (error) {
      std::string msg =
          "Invalid library proguard config parameters (these parameters are "
          "either invalid or only supported in android_binary rules): [";
      for (size_t i = 0; i < invalid_configs.size(); ++i) {
        if (i > 0) {
          msg += ", ";
        }
        msg += "'" + invalid_configs[i] + "'";
      }
      msg += ']';
      *error = msg;
    }
    return false;
  }

  std::ofstream output_file(outconfig_path_, std::ios::trunc);
  if (!output_file.is_open()) {
    if (error) {
      *error = "Unable to open output config file: " + outconfig_path_;
    }
    return false;
  }
  output_file << "# Merged from " << config_path_ << " \n" << config_string;
  if (!output_file.good()) {
    if (error) {
      *error = "Failed to write to output config file: " + outconfig_path_;
    }
    return false;
  }

  return true;
}

bool ProguardConfigValidator::ParseCommandLine(int argc, char** argv,
                                               std::string* path,
                                               std::string* output) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--path") {
      if (i + 1 < argc) {
        *path = argv[++i];
      } else {
        return false;
      }
    } else if (arg.rfind("--path=", 0) == 0) {
      *path = arg.substr(7);
    } else if (arg == "--output") {
      if (i + 1 < argc) {
        *output = argv[++i];
      } else {
        return false;
      }
    } else if (arg.rfind("--output=", 0) == 0) {
      *output = arg.substr(9);
    }
  }
  return !path->empty() && !output->empty();
}

}  // namespace proguard_allowlister
