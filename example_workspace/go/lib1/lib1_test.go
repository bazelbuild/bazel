package lib1

import (
	"testing"
)

func TestFib(t *testing.T) {
	got := Fib(5)
	want := 8

	if got != want {
		t.Fatalf("got %d want %d", got, want)
	}
}
