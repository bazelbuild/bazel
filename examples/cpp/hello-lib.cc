#include "hello-lib.h"

#include <iostream>

using std::cout;
using std::endl;
using std::string;

namespace hello {

HelloLib::HelloLib(const string& greeting) : greeting_(new string(greeting)) {
}

void HelloLib::greet(const string& thing) {
  cout << *greeting_ << " " << thing << endl;
}

}  // namespace hello
