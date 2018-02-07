import json
import sys
from urllib.parse import urlparse

def main(bep_path):
  raw_data = ""
  with open(bep_path) as f:
    raw_data = f.read()
  decoder = json.JSONDecoder()

  pos = 0
  while pos < len(raw_data):
    json_dict, size = decoder.raw_decode(raw_data[pos:])
    if "testResult" in json_dict:
      test_result = json_dict["testResult"]
      if test_result["status"] != "PASSED":
        outputs = test_result["testActionOutput"]
        for output in outputs:
          if output["name"] == "test.log":
            print(urlparse(output["uri"]).path)
    pos += size + 1

if __name__ == '__main__':
  main(sys.argv[1])
