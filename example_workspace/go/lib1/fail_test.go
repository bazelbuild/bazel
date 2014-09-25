package fail

import (
	"testing"
)

func TestFail(t *testing.T) {
	t.Fatal("I am failing.")
}
