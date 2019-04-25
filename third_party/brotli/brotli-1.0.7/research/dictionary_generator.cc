#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "./deorummolae.h"
#include "./durchschlag.h"
#include "./sieve.h"

#define METHOD_DM 0
#define METHOD_SIEVE 1
#define METHOD_DURCHSCHLAG 2
#define METHOD_DISTILL 3
#define METHOD_PURIFY 4

static size_t readInt(const char* str) {
  size_t result = 0;
  if (str[0] == 0 || str[0] == '0') {
    return 0;
  }
  for (size_t i = 0; i < 13; ++i) {
    if (str[i] == 0) {
      return result;
    }
    if (str[i] == 'k' || str[i] == 'K') {
      if ((str[i + 1] == 0) && ((result << 10) > result)) {
        return result << 10;
      }
      return 0;
    }
    if (str[i] == 'm' || str[i] == 'M') {
      if ((str[i + 1] == 0) && ((result << 20) > result)) {
        return result << 20;
      }
      return 0;
    }
    if (str[i] < '0' || str[i] > '9') {
      return 0;
    }
    size_t next = (10 * result) + (str[i] - '0');
    if (next <= result) {
      return 0;
    }
    result = next;
  }
  return 0;
}

static std::string readFile(const std::string& path) {
  std::ifstream file(path);
  std::string content(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  return content;
}

static void writeFile(const char* file, const std::string& content) {
  std::ofstream outfile(file, std::ofstream::binary);
  outfile.write(content.c_str(), static_cast<std::streamsize>(content.size()));
  outfile.close();
}

static void writeSamples(char const* argv[], const std::vector<int>& pathArgs,
    const std::vector<size_t>& sizes, const uint8_t* data) {
  size_t offset = 0;
  for (size_t i = 0; i < pathArgs.size(); ++i) {
    int j = pathArgs[i];
    const char* file = argv[j];
    size_t sampleSize = sizes[i];
    std::ofstream outfile(file, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(data + offset),
        static_cast<std::streamsize>(sampleSize));
    outfile.close();
    offset += sampleSize;
  }
}

/* Returns "base file name" or its tail, if it contains '/' or '\'. */
static const char* fileName(const char* path) {
  const char* separator_position = strrchr(path, '/');
  if (separator_position) path = separator_position + 1;
  separator_position = strrchr(path, '\\');
  if (separator_position) path = separator_position + 1;
  return path;
}

static void printHelp(const char* name) {
  fprintf(stderr, "Usage: %s [OPTION]... DICTIONARY [SAMPLE]...\n", name);
  fprintf(stderr,
      "Options:\n"
      "  --dm       use 'deorummolae' engine\n"
      "  --distill  rewrite samples; unique text parts are removed\n"
      "  --dsh      use 'durchschlag' engine (default)\n"
      "  --purify   rewrite samples; unique text parts are zeroed out\n"
      "  --sieve    use 'sieve' engine\n"
      "  -b#        set block length for 'durchschlag'; default: 1024\n"
      "  -s#        set slice length for 'distill', 'durchschlag', 'purify'\n"
      "             and 'sieve'; default: 16\n"
      "  -t#        set target dictionary size (limit); default: 16K\n"
      "  -u#        set minimum slice population (for rewrites); default: 2\n"
      "# is a decimal number with optional k/K/m/M suffix.\n"
      "WARNING: 'distill' and 'purify' will overwrite original samples!\n"
      "         Completely unique samples might become empty files.\n\n");
}

int main(int argc, char const* argv[]) {
  int dictionaryArg = -1;
  int method = METHOD_DURCHSCHLAG;
  size_t sliceLen = 16;
  size_t targetSize = 16 << 10;
  size_t blockSize = 1024;
  size_t minimumPopulation = 2;

  std::vector<uint8_t> data;
  std::vector<size_t> sizes;
  std::vector<int> pathArgs;
  size_t total = 0;
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == nullptr) {
      continue;
    }
    if (argv[i][0] == '-') {
      if (argv[i][1] == '-') {
        if (dictionaryArg != -1) {
          fprintf(stderr,
              "Method should be specified before dictionary / sample '%s'\n",
              argv[i]);
          exit(1);
        }
        if (std::strcmp("--sieve", argv[i]) == 0) {
          method = METHOD_SIEVE;
          continue;
        }
        if (std::strcmp("--dm", argv[i]) == 0) {
          method = METHOD_DM;
          continue;
        }
        if (std::strcmp("--dsh", argv[i]) == 0) {
          method = METHOD_DURCHSCHLAG;
          continue;
        }
        if (std::strcmp("--distill", argv[i]) == 0) {
          method = METHOD_DISTILL;
          continue;
        }
        if (std::strcmp("--purify", argv[i]) == 0) {
          method = METHOD_PURIFY;
          continue;
        }
        printHelp(fileName(argv[0]));
        fprintf(stderr, "Invalid option '%s'\n", argv[i]);
        exit(1);
      }
      if (argv[i][1] == 'b') {
        blockSize = readInt(&argv[i][2]);
        if (blockSize < 16 || blockSize > 65536) {
          printHelp(fileName(argv[0]));
          fprintf(stderr, "Invalid option '%s'\n", argv[i]);
          exit(1);
        }
      } else if (argv[i][1] == 's') {
        sliceLen = readInt(&argv[i][2]);
        if (sliceLen < 4 || sliceLen > 256) {
          printHelp(fileName(argv[0]));
          fprintf(stderr, "Invalid option '%s'\n", argv[i]);
          exit(1);
        }
      } else if (argv[i][1] == 't') {
        targetSize = readInt(&argv[i][2]);
        if (targetSize < 256 || targetSize > (1 << 25)) {
          printHelp(fileName(argv[0]));
          fprintf(stderr, "Invalid option '%s'\n", argv[i]);
          exit(1);
        }
      } else if (argv[i][1] == 'u') {
        minimumPopulation = readInt(&argv[i][2]);
        if (minimumPopulation < 256 || minimumPopulation > 65536) {
          printHelp(fileName(argv[0]));
          fprintf(stderr, "Invalid option '%s'\n", argv[i]);
          exit(1);
        }
      } else {
        printHelp(fileName(argv[0]));
        fprintf(stderr, "Unrecognized option '%s'\n", argv[i]);
        exit(1);
      }
      continue;
    }
    if (dictionaryArg == -1) {
      if (method != METHOD_DISTILL && method != METHOD_PURIFY) {
        dictionaryArg = i;
        continue;
      }
    }
    std::string content = readFile(argv[i]);
    data.insert(data.end(), content.begin(), content.end());
    total += content.size();
    pathArgs.push_back(i);
    sizes.push_back(content.size());
  }
  bool wantDictionary = (dictionaryArg == -1);
  if (method == METHOD_DISTILL || method == METHOD_PURIFY) {
    wantDictionary = false;
  }
  if (wantDictionary || total == 0) {
    printHelp(fileName(argv[0]));
    fprintf(stderr, "Not enough arguments\n");
    exit(1);
  }

  if (method == METHOD_SIEVE) {
    writeFile(argv[dictionaryArg], sieve_generate(
        targetSize, sliceLen, sizes, data.data()));
  } else if (method == METHOD_DM) {
    writeFile(argv[dictionaryArg], DM_generate(
        targetSize, sizes, data.data()));
  } else if (method == METHOD_DURCHSCHLAG) {
    writeFile(argv[dictionaryArg], durchschlag_generate(
        targetSize, sliceLen, blockSize, sizes, data.data()));
  } else if (method == METHOD_DISTILL) {
    durchschlag_distill(sliceLen, minimumPopulation, &sizes, data.data());
    writeSamples(argv, pathArgs, sizes, data.data());
  } else if (method == METHOD_PURIFY) {
    durchschlag_purify(sliceLen, minimumPopulation, sizes, data.data());
    writeSamples(argv, pathArgs, sizes, data.data());
  } else {
    printHelp(fileName(argv[0]));
    fprintf(stderr, "Unknown generator\n");
    exit(1);
  }
  return 0;
}
