package lib

import (
	"testing"
)

func TestMeaning(t *testing.T) {
	if m := Meaning(); m != 42 {
		t.Errorf("got %d, want 42", m)
	}
}
