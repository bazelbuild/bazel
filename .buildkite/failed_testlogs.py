import json
import sys
from urllib.parse import urlparse

def main(bep_path):
  with open(bep_path, "r") as f:
    for key, test_result in json.read(f).items():
      if key != "testResult":
        continue
      if test_result["status"] != "PASSED":
        for output in test_result["testActionOutput"]:
          if output["name"] == "test.log":
            print(urlparse(output["uri"]).path)

if __name__ == '__main__':
  main(sys.argv[1])
