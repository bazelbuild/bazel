# exit immediately if a command or pipeline fails, unless it is in a test expression
set -e

# treat unset variables as errors
set -u

# exit code of a pipeline is 0, or the non-zero exit code of the rightmost failing command
set -o pipefail
